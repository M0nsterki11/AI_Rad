"""Separate opt-in experiment. Default is a read-only preflight, never training.

Uses existing multipage train/validation split and a new exclusive output folder.
Does not load ImageNet over the network, rebuild data, or evaluate the test split.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from scripts.diagnose_models import sha, dump, code_record, write_csv
from src import train_resnet as original
from src.multipage_training import load_multipage_training_rows, make_document_balanced_sampler


def build_model(config):
    initial = ROOT / config["initial_checkpoint"]
    if sha(initial) != config["initial_checkpoint_sha256"]:
        raise ValueError("Initial checkpoint fingerprint differs from the reviewed configuration")
    checkpoint = torch.load(initial, map_location="cpu", weights_only=True)
    if checkpoint["class_names"] != original.CLASS_NAMES:
        raise ValueError("Initial checkpoint class order differs")
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(original.CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(("layer4.", "fc."))
    return model


def training_mode_with_frozen_bn(model):
    model.train()
    # eval freezes running_mean/running_var/counter, not layer4 affine gradients.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/resnet50_layer4_proposal.json")
    parser.add_argument("--train", action="store_true", help="Explicitly start this separate experiment")
    parser.add_argument("--preflight-report", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if (config["trainable_prefixes"] != ["layer4.", "fc."] or
        config["batchnorm_running_statistics"] != "frozen_all_layers" or
        config["batchnorm_affine_parameters"] != "trainable_only_in_layer4" or
        config["selection_metric"] != "validation_macro_f1" or config["aggregation_top_k"] != 3):
        parser.error("Unsupported training/BatchNorm/selection policy")
    output = (ROOT / config["output_dir"]).resolve()
    model_root = (ROOT / "models").resolve()
    if not output.is_relative_to(model_root) or output == model_root or output.exists():
        parser.error("Output must be a NEW subdirectory inside models; no overwrite allowed")
    if config["backbone_learning_rate"] >= config["head_learning_rate"]:
        parser.error("Backbone learning rate must be smaller than head learning rate")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    documents, splits, page_path = load_multipage_training_rows("resnet_page")
    model = build_model(config)
    training_mode_with_frozen_bn(model)
    report = {"config": config, "training_started": bool(args.train), "documents": len(documents),
        "document_manifest_sha256": sha(ROOT / "data/multipage/document_manifest.csv"),
        "page_manifest_sha256": sha(page_path), "code": code_record(),
        "runner_sha256": sha(Path(__file__)),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_names": [n for n, p in model.named_parameters() if p.requires_grad],
        "batchnorm_all_eval": all(not m.training for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)),
        "split_page_counts": {k: len(v) for k, v in splits.items()}, "test_forward_passes": 0}
    if args.preflight_report:
        destination = args.preflight_report.resolve()
        if not destination.is_relative_to((ROOT / "reports").resolve()):
            parser.error("Preflight report must be inside reports")
        dump(destination, report)
    if not args.train:
        print(json.dumps({k: report[k] for k in ["training_started", "trainable_parameters", "total_parameters", "batchnorm_all_eval", "test_forward_passes"]}))
        return
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "run_config.json", report)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("layer4.")], "lr": config["backbone_learning_rate"]},
        {"params": list(model.fc.parameters()), "lr": config["head_learning_rate"]},
    ], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    transform = original.make_transforms()
    train_loader = DataLoader(original.DocumentPageDataset(splits["train"], transform), batch_size=config["batch_size"],
                              sampler=make_document_balanced_sampler(splits["train"]), num_workers=0)
    validation_loader = DataLoader(original.DocumentPageDataset(splits["validation"], transform), batch_size=config["batch_size"], shuffle=False, num_workers=0)
    best, stale, history = -1.0, 0, []
    for epoch in range(1, config["epochs"] + 1):
        training_mode_with_frozen_bn(model)
        total_loss, count = 0., 0
        for images, labels, _, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            count += len(labels)
        raw = original.collect_evaluation(model, validation_loader, criterion, device)
        # Fixed aggregation from existing validation-selected ResNet configuration.
        metrics, comparisons = original.aggregate_raw(raw, method=config["validation_aggregation"])
        row = dict(epoch=epoch, train_loss=total_loss/count, validation_loss=metrics["loss"],
                   validation_macro_f1=metrics["macro_f1"], validation_accuracy=metrics["accuracy"])
        history.append(row)
        dump(output / f"epoch_{epoch:02d}.json", row)
        print(json.dumps(row), flush=True)
        if metrics["macro_f1"] > best:
            best, stale = float(metrics["macro_f1"]), 0
            original.save_checkpoint(model, output / "best_model.pth", epoch, metrics, comparisons)
        else:
            stale += 1
        if stale >= config["early_stopping_patience"]:
            break
    write_csv(output / "training_history.csv", history)
    print(f"Saved new experiment only: {output}. Test split was not evaluated.")


if __name__ == "__main__":
    main()
