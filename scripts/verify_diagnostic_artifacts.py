"""Final invariant checks, including serial preparation versus retained results."""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_models import sha, rows, dump, MODELS, code_record
from PIL import Image


def prepared_signature(run, document_id):
    record = json.loads((run / "prepared" / document_id / "preparation.json").read_text(encoding="utf-8"))
    prep = record.get("prepared", {})
    pages = []
    for page in prep.get("page_artifacts", []):
        with Image.open(page["image_path"]) as image:
            pixel_hash = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
            size = list(image.size)
        pages.append(dict(page_index=page["page_index"], image_sha256=pixel_hash, size=size,
            words_boxes_sha256=hashlib.sha256(json.dumps([page.get("words"),page.get("boxes")],ensure_ascii=True).encode()).hexdigest()))
    return dict(pages=pages, text_sha256=sha(prep["text_path"]) if prep.get("text_path") else None,
                total_pages=prep.get("total_pages"), layout_page_indices=prep.get("layout_page_indices"), errors=prep.get("errors"),
                preparation_error=record.get("preparation_error"))


def main():
    reports = ROOT / "reports"
    baseline, after, serial = [reports / n for n in ["baseline_20260915", "after_20260915", "serial_verification_20260915"]]
    selection = json.loads((after / "selection.json").read_text(encoding="utf-8"))
    assert selection == json.loads((baseline / "selection.json").read_text(encoding="utf-8"))
    assert selection == json.loads((serial / "selection.json").read_text(encoding="utf-8"))
    prepared_checks = [{"id": d["id"], "identical": prepared_signature(after,d["id"]) == prepared_signature(serial,d["id"])} for d in selection["documents"]]
    assert all(d["identical"] for d in prepared_checks), "Serial preparation differs; investigate before reporting"
    audit = json.loads((baseline / "dataset_audit.json").read_text(encoding="utf-8"))
    manifest_hashes = {"document_manifest.csv": audit["manifest_sha256"],
        "page_manifest.csv": audit["artifacts"]["resnet_page"]["sha256"], "chunk_manifest.jsonl": audit["artifacts"]["chunk"]["sha256"]}
    unchanged = {name: sha(ROOT / "data/multipage" / name) == expected for name, expected in manifest_hashes.items()}
    assert all(unchanged.values())
    documents = rows(ROOT / "data/multipage/document_manifest.csv")
    expected = {d["document_id"]:d["label"] for d in documents if d["split"]=="test"}
    models = {}
    for key in MODELS:
        old = json.loads((baseline/key/"runtime_model.json").read_text(encoding="utf-8"))
        current = json.loads((after/key/"runtime_model.json").read_text(encoding="utf-8"))
        equal = sha(Path(current["resolved_weight_path"])) == current["weight_sha256"] == old["weight_sha256"]
        assert equal and current["all_loaded_tensors_match_saved"]
        predictions = rows(ROOT/"results"/(key+"_multipage")/"test_predictions.csv")
        models[key] = dict(weights_unchanged=equal, historical_test_rows=len(predictions),
            historical_test_ids_match_current={p["document_id"] for p in predictions}==set(expected),
            historical_test_label_mismatches=sum(p["true_label"]!=expected.get(p["document_id"]) for p in predictions))
    local_links = re.findall(r'\]\(([^)]+)\)', (reports/"diagnostika_modela.md").read_text(encoding="utf-8"))
    missing_links = [link for link in local_links if not link.startswith(("https:","http:","#")) and not (reports/link).exists()]
    assert not missing_links, missing_links
    result = dict(code=code_record(), serial_preparation_matches=prepared_checks, unchanged_manifests=unchanged,
                  models=models, missing_report_links=missing_links,
                  future_experiment_directory_exists=(ROOT/"models/resnet50_layer4_experiment_20260915").exists())
    assert not result["future_experiment_directory_exists"]
    dump(reports/"final_verification.json", result)
    print(json.dumps(dict(serial_preparation_matches=sum(d["identical"] for d in prepared_checks),
                         unchanged_manifests=unchanged, models=models, missing_report_links=missing_links),indent=2))


if __name__ == "__main__":
    main()
