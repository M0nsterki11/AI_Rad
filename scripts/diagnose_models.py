"""Read-only model/data diagnostics. Writes only NEW files in a reports run directory.

No training, downloading, split changes or edits to existing checkpoints. Run phases
in order: prepare, evaluate, audit, tests. Reuse --selection for before/after runs.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
LABELS = ["invoice", "cv", "contract", "email", "scientific"]
MODELS = ["resnet50", "xlm_roberta", "layoutlmv3"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=str)


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, values, fields=None):
    fields = fields or list(values[0])
    with Path(path).open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def resolve(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def code_record():
    result = {"utc": datetime.now(timezone.utc).isoformat(), "python": sys.version}
    for arg, key in [("HEAD", "commit"), ("--abbrev-ref HEAD", "branch")]:
        result[key] = subprocess.check_output(["git", "rev-parse", *arg.split()], cwd=ROOT, text=True).strip()
    result["code_sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in
        [ROOT / "app.py", *sorted((ROOT / "src").glob("*.py")), Path(__file__)]}
    return result


def select(args):
    if args.selection:
        return json.loads(resolve(args.selection).read_text(encoding="utf-8"))
    docs = rows(ROOT / "data/multipage/document_manifest.csv")
    rng = random.Random(args.seed)
    selected = []
    for label in LABELS:
        eligible = sorted((d for d in docs if d["split"] == "test" and d["label"] == label), key=lambda d: d["document_id"])
        for doc in rng.sample(eligible, min(args.per_class, len(eligible))):
            selected.append(dict(dataset="internal_test", id=doc["document_id"], label=label,
                                 raw_path=doc["raw_path"], label_source="multipage_manifest"))
    for p in sorted((ROOT / "data/external_test/raw").rglob("*.pdf")):
        label = next(part for part in p.relative_to(ROOT / "data/external_test/raw").parts[:-1] if part in LABELS)
        selected.append(dict(dataset="external_existing", id=p.stem, label=label,
                             raw_path=str(p.relative_to(ROOT)), label_source="folder_not_independently_adjudicated"))
    return {"seed": args.seed, "per_class": args.per_class,
            "selection_before_predictions": True, "documents": selected}


def prepare(args):
    from src.document_adapter import prepare_document_for_models
    from src.preprocess import TESSERACT_AVAILABLE
    import pytesseract
    selection = select(args)
    dump(args.output / "selection.json", selection)
    dump(args.output / "code.json", code_record())
    dump(args.output / "ocr_runtime.json", {"available": TESSERACT_AVAILABLE,
        "command": pytesseract.pytesseract.tesseract_cmd,
        "version": str(pytesseract.get_tesseract_version()) if TESSERACT_AVAILABLE else None,
        "languages": pytesseract.get_languages(config="") if TESSERACT_AVAILABLE else [],
        "requested_ocr_language": "no lang parameter (Tesseract default)"})
    def one(doc):
        started = time.perf_counter()
        directory = args.output / "prepared" / doc["id"]
        directory.mkdir(parents=True, exist_ok=False)
        record = dict(doc, raw_sha256=sha(resolve(doc["raw_path"])))
        try:
            record["prepared"] = prepare_document_for_models(resolve(doc["raw_path"]), directory)
        except Exception as exc:
            record["preparation_error"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = time.perf_counter() - started
        dump(directory / "preparation.json", record)
        print(f"PREP {doc['dataset']} {doc['id']} {record['seconds']:.1f}s", flush=True)
        return {k: v for k, v in record.items() if k != "prepared"}
    # PyMuPDF is not thread-safe; PDF preparation must not share this process's threads.
    summary = [one(doc) for doc in selection["documents"]]
    dump(args.output / "preparation_summary.json", summary)


class Capture:
    """Hash canonical per-item inputs; store logits, never document text in logs."""
    def __init__(self, model):
        self.items = []
        self.hook = model.register_forward_hook(self.record, with_kwargs=True)

    def record(self, module, positional, keyword, output):
        import torch
        inputs = {k: v for k, v in keyword.items() if isinstance(v, torch.Tensor) and k != "labels"}
        if positional:
            inputs["pixel_values"] = positional[0]
        logits = output if isinstance(output, torch.Tensor) else output.logits
        for i, logit in enumerate(logits.detach().float().cpu()):
            hashes, shapes = {}, {}
            length = int(inputs["attention_mask"][i].sum()) if "attention_mask" in inputs else None
            for k, v in inputs.items():
                tensor = v[i].detach().cpu().contiguous()
                if length is not None and k in {"input_ids", "attention_mask", "token_type_ids", "bbox"}:
                    tensor = tensor[:length].contiguous()
                hashes[k] = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
                shapes[k] = list(tensor.shape)
            self.items.append({"input_sha256": hashes, "input_shapes": shapes, "logits": logit.tolist()})

    def close(self):
        self.hook.remove()


def model_record(key, loaded):
    import torch
    from safetensors import safe_open
    from src import predict_resnet, predict_text_model, predict_layoutlm
    model = loaded[0]
    path = (predict_resnet.active_model_path() if key == "resnet50" else
            (predict_text_model.active_model_dir() if key == "xlm_roberta" else predict_layoutlm.active_model_dir()) / "model.safetensors")
    state = model.state_dict()
    if key == "resnet50":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        saved = checkpoint["model_state_dict"]
        keys = set(saved)
        unequal = [k for k in keys & state.keys() if not torch.equal(saved[k], state[k].detach().cpu())]
        metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    else:
        with safe_open(path, framework="pt", device="cpu") as saved:
            keys = set(saved.keys())
            unequal = [k for k in keys & state.keys() if not torch.equal(saved.get_tensor(k), state[k].detach().cpu())]
        metadata = model.config.to_dict()
    return {"model": key, "resolved_weight_path": str(path.resolve()), "weight_sha256": sha(path),
            "weight_bytes": path.stat().st_size, "eval_mode": not model.training,
            "loaded_state_keys": len(state), "saved_state_keys": len(keys),
            "missing_saved_keys": sorted(set(state) - keys), "unused_saved_keys": sorted(keys - set(state)),
            "unequal_loaded_tensors": unequal, "all_loaded_tensors_match_saved": keys == set(state) and not unequal,
            "aggregation": [model.document_aggregation_method, model.document_aggregation_top_k],
            "metadata": metadata}


def stored_prepared(doc, page_rows):
    pages = [dict(p, image_path=str(resolve(p["image_path"])), ocr_path=str(resolve(p["ocr_path"])),
                  page_index=int(p["page_index"])) for p in page_rows]
    for page in pages:
        payload = json.loads(Path(page["ocr_path"]).read_text(encoding="utf-8"))
        page["words"] = payload.get("words", [])
        page["boxes"] = payload.get("boxes", [])
    return {"image_path": pages[0]["image_path"] if pages else None,
            "ocr_path": pages[0]["ocr_path"] if pages else None,
            "text_path": str(resolve(doc["text_path"])), "page_artifacts": pages,
            "layout_page_artifacts": [p for p in pages if p["layout_status"] == "valid"],
            "total_pages": int(pages[0]["total_pages"]) if pages else 0, "errors": []}


def reference(key, loaded, page_rows, chunks):
    import torch
    from torch.utils.data import DataLoader
    from src import train_resnet, train_text_model, train_layoutlm
    from src.multipage import aggregate_scores
    model = loaded[0]
    device = next(model.parameters()).device
    if key == "resnet50":
        dataset = train_resnet.DocumentPageDataset(page_rows, train_resnet.make_transforms())
        raw = train_resnet.collect_evaluation(model, DataLoader(dataset, batch_size=8), torch.nn.CrossEntropyLoss(), device)
    elif key == "xlm_roberta":
        dataset = train_text_model.DocumentChunkDataset(chunks)
        loader = DataLoader(dataset, batch_size=4, collate_fn=train_text_model.ChunkCollator(loaded[1]))
        raw = train_text_model.collect_evaluation(model, loader, device)
    else:
        valid = [p for p in page_rows if p["layout_status"] == "valid"]
        dataset = train_layoutlm.DocumentPageDataset(valid, loaded[1], 512)
        raw = train_layoutlm.collect_evaluation(model, DataLoader(dataset, batch_size=2), device)
    logits = next(iter(raw["logits_by_document"].values()))
    _, probs = aggregate_scores(logits, method=model.document_aggregation_method,
                                top_k=model.document_aggregation_top_k, scores_are_logits=True)
    return {"predicted_class": LABELS[int(probs.argmax())], "confidence": float(probs.max()),
            "probabilities": {c: float(probs[i]) for i, c in enumerate(LABELS)}}


def compare_captures(left, right):
    import numpy as np
    equal = len(left) == len(right) and all(a["input_sha256"] == b["input_sha256"] for a, b in zip(left, right))
    delta = float(np.max(np.abs(np.array([v["logits"] for v in left]) - np.array([v["logits"] for v in right])))) if left and len(left) == len(right) else None
    return {"items_left": len(left), "items_right": len(right), "canonical_inputs_equal": equal, "max_abs_logit_difference": delta}


def metrics(records):
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    result = {}
    for dataset in sorted({r["dataset"] for r in records}):
        for pathway in sorted({r["pathway"] for r in records if r["dataset"] == dataset}):
            subset = [r for r in records if r["dataset"] == dataset and r["pathway"] == pathway]
            true = [r["true_label"] for r in subset]
            pred = [r["predicted_label"] or "PROCESSING_ERROR" for r in subset]
            report = classification_report(true, pred, labels=LABELS, output_dict=True, zero_division=0)
            accepted = [r for r in subset if r["confidence"] is not None and r["confidence"] >= .8]
            result[f"{dataset}/{pathway}"] = {"n": len(subset), "accuracy": accuracy_score(true, pred),
                "macro_f1": report["macro avg"]["f1-score"], "per_class": {c: report[c] for c in LABELS},
                "confusion_labels": LABELS + ["PROCESSING_ERROR"],
                "confusion_matrix": confusion_matrix(true, pred, labels=LABELS+["PROCESSING_ERROR"]).tolist(),
                "processing_errors": sum(p == "PROCESSING_ERROR" for p in pred),
                "confidence_80": {"covered": len(accepted), "coverage": len(accepted)/len(subset),
                    "wrong": sum(r["true_label"] != r["predicted_label"] for r in accepted)}}
    return result


def evaluate(args):
    import torch
    import app
    selection = json.loads((args.output / "selection.json").read_text(encoding="utf-8"))
    documents = {d["document_id"]: d for d in rows(ROOT / "data/multipage/document_manifest.csv")}
    pages, chunks = defaultdict(list), defaultdict(list)
    for row in rows(ROOT / "data/multipage/page_manifest.csv"):
        pages[row["document_id"]].append(row)
    with (ROOT / "data/multipage/chunk_manifest.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            chunks[row["document_id"]].append(row)
    getters = {"resnet50": app.get_resnet_model, "xlm_roberta": app.get_text_model, "layoutlmv3": app.get_layoutlm_model}
    for key in args.models or MODELS:
        out = args.output / args.evaluation_tag / key if args.evaluation_tag else args.output / key
        out.mkdir(parents=True, exist_ok=False)
        dump(out / "execution_code.json", code_record())
        print(f"LOAD {key}", flush=True)
        loaded = getters[key]()  # actual Streamlit resource getter, not a replacement loader
        dump(out / "runtime_model.json", model_record(key, loaded))
        capture = Capture(loaded[0])
        predictions, comparisons = [], []
        with (out / "items.jsonl").open("x", encoding="utf-8") as item_file:
            for doc in selection["documents"]:
                document_id = doc["id"]
                fresh = json.loads((args.output / "prepared" / document_id / "preparation.json").read_text(encoding="utf-8"))
                paths = ["app_fresh"] + (["app_stored", "evaluation_stored"] if doc["dataset"] == "internal_test" else [])
                captured = {}
                for pathway in paths:
                    capture.items = []
                    started = time.perf_counter()
                    reason, result = "", None
                    try:
                        if pathway == "evaluation_stored":
                            result = reference(key, loaded, pages[document_id], chunks[document_id])
                        else:
                            prepared = fresh["prepared"] if pathway == "app_fresh" else stored_prepared(documents[document_id], pages[document_id])
                            outcome = app.run_prepared_model_prediction(prepared, key)
                            result, reason = outcome["result"], outcome["reason"]
                        if not result:
                            reason = reason or fresh.get("preparation_error", "No result")
                    except Exception as exc:
                        reason = f"{type(exc).__name__}: {exc}"
                    record = dict(dataset=doc["dataset"], document_id=document_id, model=key, pathway=pathway,
                        true_label=doc["label"], predicted_label=result["predicted_class"] if result else "",
                        confidence=result["confidence"] if result else None, error=reason,
                        seconds=time.perf_counter()-started, items=len(capture.items))
                    predictions.append(record)
                    captured[pathway] = capture.items
                    item_file.write(json.dumps(dict(record, result=result, item_details=capture.items), ensure_ascii=False, default=str)+"\n")
                    item_file.flush()
                if "app_stored" in captured:
                    comparisons.append(dict(document_id=document_id,
                        evaluation_vs_app_stored=compare_captures(captured["evaluation_stored"], captured["app_stored"]),
                        app_stored_vs_fresh=compare_captures(captured["app_stored"], captured["app_fresh"])))
                print(f"PRED {key} {doc['dataset']} {document_id} {predictions[-len(paths)]['predicted_label'] or 'ERROR'}", flush=True)
        capture.close()
        write_csv(out / "predictions.csv", predictions)
        dump(out / "metrics.json", metrics(predictions))
        dump(out / "input_comparisons.json", comparisons)
        getters[key].clear()
        del loaded, capture
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def audit(args):
    from src.multipage_training import load_multipage_training_rows
    docs = rows(ROOT / "data/multipage/document_manifest.csv")
    page_rows = rows(ROOT / "data/multipage/page_manifest.csv")
    result = {"manifest_sha256": sha(ROOT / "data/multipage/document_manifest.csv"), "counts": {}, "artifacts": {}}
    for label in LABELS:
        result["counts"][label] = dict(Counter(d["split"] for d in docs if d["label"] == label))
    for kind in ["resnet_page", "layout_page", "chunk"]:
        selected, splits, path = load_multipage_training_rows(kind)
        result["artifacts"][kind] = {"documents": len(selected), "counts": {k: len(v) for k, v in splits.items()}, "sha256": sha(path)}
    old = {}
    for split in ["train", "validation", "test"]:
        for row in rows(ROOT / "data/splits" / (split + ".csv")):
            old[row.get("id", row.get("document_id"))] = split
    disagreements = [dict(id=d["document_id"], legacy=old.get(d["document_id"]), multipage=d["split"]) for d in docs if old.get(d["document_id"]) != d["split"]]
    write_csv(args.output / "split_disagreements.csv", disagreements)
    result["legacy_disagreements"] = len(disagreements)
    sources = rows(ROOT / "data/dataset_sources_extra.csv")
    result["source_counts"] = dict(Counter(r["source_name"] for r in sources))
    result["source_language_counts"] = dict(Counter(r["language"] or "unknown" for r in sources))
    result["source_synthetic_counts"] = dict(Counter(r["is_synthetic"] for r in sources))
    result["source_augmented_counts"] = dict(Counter(r["is_augmented"] for r in sources))
    result["source_missing_ids"] = len({d["document_id"] for d in docs} - {s["id"] for s in sources})
    groups = {k: defaultdict(list) for k in ["raw_file", "page_file", "text_normalized", "text_digits_removed"]}
    raw_mismatch, failures, keywords = [], [], []
    def hash_doc(d):
        try:
            digest = sha(resolve(d["raw_path"]))
            text = resolve(d["text_path"]).read_text(encoding="utf-8", errors="ignore").casefold()
            normalized = " ".join(re.findall(r"\w+", text))
            template = re.sub(r"\d+", "#", normalized)
            flags = [term for term in ["europass", "uplatnica", "hrvatski telekom", "gmail", "outlook"] if term in text]
            return d, digest, hashlib.sha256(normalized.encode()).hexdigest() if normalized else None, hashlib.sha256(template.encode()).hexdigest() if len(template) > 100 else None, flags, None
        except Exception as exc:
            return d, None, None, None, [], f"{type(exc).__name__}: {exc}"
    with ThreadPoolExecutor(max_workers=4) as pool:
        for d, digest, normalized, template, flags, error in pool.map(hash_doc, docs):
            if error:
                failures.append(dict(id=d["document_id"], error=error))
                continue
            entry = {k: d[k] for k in ["document_id", "label", "split"]}
            groups["raw_file"][digest].append(entry)
            if normalized:
                groups["text_normalized"][normalized].append(entry)
            if template:
                groups["text_digits_removed"][template].append(entry)
            if digest != d["raw_sha256"]:
                raw_mismatch.append(d["document_id"])
            if flags:
                keywords.append(dict(entry, flags=flags))
    print("AUDIT raw/text complete", flush=True)
    def page_hash(p):
        try:
            return p, sha(resolve(p["image_path"])), None
        except Exception as exc:
            return p, None, str(exc)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for p, digest, error in pool.map(page_hash, page_rows):
            if error:
                failures.append(dict(id=p["document_id"], page=p["page_index"], error=error))
            else:
                groups["page_file"][digest].append({k: p[k] for k in ["document_id", "page_index", "label", "split"]})
    duplicates = {}
    result["duplicate_summary"] = {}
    for kind, hashes in groups.items():
        matches = [dict(sha256=digest, entries=entries, cross_split=len({e["split"] for e in entries}) > 1)
                   for digest, entries in hashes.items() if len({e["document_id"] for e in entries}) > 1]
        duplicates[kind] = matches
        result["duplicate_summary"][kind] = {"groups": len(matches), "cross_split_groups": sum(m["cross_split"] for m in matches)}
    selection = json.loads((args.output / "selection.json").read_text(encoding="utf-8"))
    result["external_exact_raw_overlap"] = [{"id": d["id"], "matches": groups["raw_file"].get(sha(resolve(d["raw_path"])), [])}
        for d in selection["documents"] if d["dataset"] == "external_existing"]
    result["raw_manifest_hash_mismatches"] = raw_mismatch
    result["read_errors"] = failures
    result["keyword_counts_not_verified_representation"] = dict(Counter(term for d in keywords for term in d["flags"]))
    dump(args.output / "dataset_audit.json", result)
    dump(args.output / "duplicate_groups.json", duplicates)
    dump(args.output / "keyword_candidates.json", keywords)


def run_tests(args):
    result = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
                            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    with (args.output / "tests.txt").open("x", encoding="utf-8") as f:
        f.write(result.stdout + "\n" + result.stderr)
    dump(args.output / "test_status.json", {"returncode": result.returncode})
    print(result.stderr[-1500:], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "evaluate", "audit", "tests"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument("--evaluation-tag", help="New evaluation subdirectory; preserves earlier runs")
    args = parser.parse_args()
    if args.evaluation_tag and (Path(args.evaluation_tag).name != args.evaluation_tag or args.evaluation_tag in {'.', '..'}):
        parser.error("Evaluation tag must be a plain directory name")
    args.output = resolve(args.output).resolve()
    if not args.output.is_relative_to((ROOT / "reports").resolve()):
        parser.error("Output must be a run subdirectory of reports")
    if args.output == (ROOT / "reports").resolve():
        parser.error("Choose a separate run subdirectory")
    if args.phase == "prepare":
        args.output.mkdir(parents=True, exist_ok=False)
    elif not args.output.is_dir():
        parser.error("Run prepare first")
    {"prepare": prepare, "evaluate": evaluate, "audit": audit, "tests": run_tests}[args.phase](args)


if __name__ == "__main__":
    main()
