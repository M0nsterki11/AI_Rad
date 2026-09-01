from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

try:
    from .multipage import aggregate_scores
    from .multipage_preprocess import prepare_inference_file_artifacts
    from .multipage_training import load_aggregation_config
except ImportError:
    from multipage import aggregate_scores  # type: ignore
    from multipage_preprocess import prepare_inference_file_artifacts  # type: ignore
    from multipage_training import load_aggregation_config  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MULTIPAGE_MODEL_PATH = PROJECT_ROOT / "models" / "resnet50_multipage" / "best_model.pth"
LEGACY_MODEL_PATH = PROJECT_ROOT / "models" / "resnet50" / "best_model.pth"
MODEL_PATH = LEGACY_MODEL_PATH
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}


def active_model_path():
    return MULTIPAGE_MODEL_PATH if MULTIPAGE_MODEL_PATH.is_file() else LEGACY_MODEL_PATH


def load_label_mapping(model_path=None):
    model_path = Path(model_path or active_model_path())
    mapping_path = model_path.parent / "label_mapping.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Missing label mapping: {mapping_path}")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    class_names = mapping.get("class_names")
    label_to_index = mapping.get("label_to_index")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("label_mapping.json must contain class_names.")
    if not isinstance(label_to_index, dict):
        raise ValueError("label_mapping.json must contain label_to_index.")
    ordered = [
        label for label, _ in sorted(label_to_index.items(), key=lambda item: int(item[1]))
    ]
    if ordered != class_names:
        raise ValueError("class_names order does not match label_to_index.")
    return class_names, label_to_index


def inspect_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        structure = {
            "type": "training_checkpoint",
            "keys": sorted(checkpoint.keys()),
            "epoch": checkpoint.get("epoch"),
            "validation_macro_f1": checkpoint.get("validation_macro_f1"),
            "checkpoint_class_names": checkpoint.get("class_names"),
            "checkpoint_label_to_index": checkpoint.get("label_to_index"),
            "aggregation_method": checkpoint.get("aggregation_method"),
            "aggregation_top_k": checkpoint.get("aggregation_top_k"),
        }
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        structure = {"type": "raw_state_dict", "keys_sample": list(checkpoint)[:10]}
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
    if "fc.weight" not in state_dict or "fc.bias" not in state_dict:
        raise ValueError("Checkpoint is missing fc.weight or fc.bias.")
    output_count = int(state_dict["fc.bias"].shape[0])
    input_count = int(state_dict["fc.weight"].shape[1])
    structure["fc_weight_shape"] = list(state_dict["fc.weight"].shape)
    structure["fc_bias_shape"] = list(state_dict["fc.bias"].shape)
    return state_dict, output_count, input_count, structure


def make_preprocess():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_model(device):
    model_path = active_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing ResNet50 checkpoint: {model_path}")
    class_names, label_to_index = load_label_mapping(model_path)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict, output_count, input_count, structure = inspect_checkpoint(checkpoint)
    if output_count != len(class_names):
        raise ValueError("Checkpoint output count does not match label mapping.")
    checkpoint_mapping = structure.get("checkpoint_label_to_index")
    if checkpoint_mapping is not None and checkpoint_mapping != label_to_index:
        raise ValueError("Checkpoint label mapping does not match label_mapping.json.")
    model = models.resnet50(weights=None)
    if input_count != model.fc.in_features:
        raise ValueError(f"Unexpected ResNet50 fc input size: {input_count}")
    model.fc = torch.nn.Linear(model.fc.in_features, output_count)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    method, top_k = load_aggregation_config(model_path.parent / "aggregation_config.json")
    if structure.get("aggregation_method"):
        method = str(structure["aggregation_method"])
        top_k = int(structure.get("aggregation_top_k") or top_k)
    model.document_aggregation_method = method
    model.document_aggregation_top_k = top_k
    structure.update(
        {
            "model_path": str(model_path),
            "aggregation_method": method,
            "aggregation_top_k": top_k,
            "multipage_checkpoint": model_path == MULTIPAGE_MODEL_PATH,
        }
    )
    return model, class_names, structure


@torch.no_grad()
def predict_images(
    images,
    *,
    page_indices=None,
    total_pages=None,
    model=None,
    class_names=None,
    device=None,
    batch_size=8,
    aggregation_method=None,
    aggregation_top_k=None,
):
    if not images:
        raise ValueError("No page images were provided for ResNet50 prediction.")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_structure = None
    if model is None or class_names is None:
        model, class_names, checkpoint_structure = load_model(device)
    method = aggregation_method or getattr(
        model, "document_aggregation_method", "top_k_mean"
    )
    top_k = int(
        aggregation_top_k or getattr(model, "document_aggregation_top_k", 3)
    )
    page_indices = list(page_indices or range(len(images)))
    if len(page_indices) != len(images):
        raise ValueError("page_indices length does not match images length.")
    preprocess = make_preprocess()
    tensors = torch.stack([preprocess(image.convert("RGB")) for image in images])
    logits_parts = []
    elapsed = 0.0
    for start_index in range(0, len(tensors), batch_size):
        batch = tensors[start_index : start_index + batch_size].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        logits = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        logits_parts.append(logits.detach().float().cpu())
    all_logits = torch.cat(logits_parts, dim=0)
    _, document_probabilities = aggregate_scores(
        all_logits, method=method, top_k=top_k, scores_are_logits=True
    )
    probability_map = {
        label: float(document_probabilities[index])
        for index, label in enumerate(class_names)
    }
    predicted_index = int(document_probabilities.argmax().item())
    page_probabilities = torch.softmax(all_logits, dim=1)
    page_predictions = []
    for position, page_index in enumerate(page_indices):
        probabilities = {
            label: float(page_probabilities[position, index])
            for index, label in enumerate(class_names)
        }
        best_index = int(page_probabilities[position].argmax().item())
        page_predictions.append(
            {
                "page_index": int(page_index),
                "predicted_class": class_names[best_index],
                "confidence": probabilities[class_names[best_index]],
                "probabilities": probabilities,
            }
        )
    return {
        "predicted_class": class_names[predicted_index],
        "confidence": probability_map[class_names[predicted_index]],
        "probabilities": probability_map,
        "prediction_time_seconds": elapsed,
        "device": str(device),
        "total_pages": int(total_pages or len(images)),
        "analyzed_page_indices": page_indices,
        "pages_analyzed": len(images),
        "page_predictions": page_predictions,
        "aggregation_method": method,
        "aggregation_top_k": top_k,
        "checkpoint_structure": checkpoint_structure,
    }


def predict_file(path, model=None, class_names=None, device=None):
    path = Path(path)
    if path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported extension '{path.suffix}'. Supported: {supported}")
    with tempfile.TemporaryDirectory(prefix="resnet_multipage_") as temporary_dir:
        total_pages, selected_indices, artifacts = prepare_inference_file_artifacts(
            path, Path(temporary_dir)
        )
        artifacts = [artifact for artifact in artifacts if artifact.image_path.is_file()]
        if not artifacts:
            raise ValueError("Document has no renderable pages for ResNet50 prediction")
        images = []
        for artifact in artifacts:
            with Image.open(artifact.image_path) as source:
                images.append(source.convert("RGB"))
        result = predict_images(
            images,
            page_indices=[artifact.page_index for artifact in artifacts],
            total_pages=total_pages,
            model=model,
            class_names=class_names,
            device=device,
        )
    result["file"] = str(path)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Predict with multi-page ResNet50.")
    parser.add_argument("--file", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, structure = load_model(device)
    result = predict_file(args.file, model=model, class_names=class_names, device=device)
    result["checkpoint_structure"] = structure
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
