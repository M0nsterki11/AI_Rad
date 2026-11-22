"""Read-only overlap, text-quality and PDF metadata checks; no classifier fitting."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_models import rows, dump, resolve, write_csv


def main():
    import fitz
    import numpy as np
    from PIL import Image
    from sklearn.feature_extraction.text import TfidfVectorizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to((ROOT / "reports").resolve()):
        parser.error("Output must be under reports")
    out.mkdir(parents=True, exist_ok=False)
    docs = rows(ROOT / "data/multipage/document_manifest.csv")
    by_id = {d["document_id"]: d for d in docs}
    texts = [resolve(d["text_path"]).read_text(encoding="utf-8", errors="ignore") for d in docs]
    result = {"diagnostic_only_no_classifier_training": True}
    result["texts_under_20_chars"] = [{"id": d["document_id"], "label": d["label"], "split": d["split"], "chars": len(t.strip())}
                                     for d, t in zip(docs, texts) if len(t.strip()) < 20]
    tokens = defaultdict(list)
    with (ROOT / "data/multipage/chunk_manifest.jsonl").open(encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            tokens[chunk["document_id"]].append((chunk["chunk_index"], chunk["input_ids"]))
    hashes = defaultdict(list)
    for document_id, values in tokens.items():
        fingerprint = hashlib.sha256(json.dumps(sorted(values)).encode()).hexdigest()
        hashes[fingerprint].append({k: by_id[document_id][k] for k in ["document_id", "label", "split"]})
    exact_groups = [dict(sha256=h, entries=entries) for h, entries in hashes.items() if len({e["split"] for e in entries}) > 1]
    result["identical_selected_chunk_inputs_cross_split_groups"] = len(exact_groups)
    result["test_ids_identical_chunk_inputs_in_train"] = sorted({e["document_id"] for g in exact_groups
        if any(e["split"] == "train" for e in g["entries"]) for e in g["entries"] if e["split"] == "test"})
    dump(out / "identical_token_groups.json", exact_groups)
    old_duplicates = json.loads((ROOT / "reports/baseline_20260915/duplicate_groups.json").read_text(encoding="utf-8"))
    page_manifest = rows(ROOT / "data/multipage/page_manifest.csv")
    page_index = {(p["document_id"], p["page_index"]): p for p in page_manifest}
    page_inspection = []
    for group in old_duplicates["page_file"]:
        entry = group["entries"][0]
        page = page_index[entry["document_id"], entry["page_index"]]
        with Image.open(resolve(page["image_path"])) as image:
            extrema = image.convert("RGB").getextrema()
            constant = all(low == high for low, high in extrema)
        page_inspection.append(dict(group, constant_color_page=constant, extrema=extrema))
    dump(out / "identical_page_inspection.json", page_inspection)
    normalized_details = []
    for group in old_duplicates["text_normalized"]:
        if not group["cross_split"]:
            continue
        entry = group["entries"][0]
        text = texts[docs.index(by_id[entry["document_id"]])]
        normalized_details.append(dict(group, normalized_chars=len(" ".join(re.findall(r"\w+", text.casefold())))))
    dump(out / "identical_normalized_text_details.json", normalized_details)
    # Candidate similarity, not a declaration that two originals are duplicates.
    # Vectorizer is used ONLY for this audit, never classifier training/evaluation.
    vectorizer = TfidfVectorizer(max_features=20000, min_df=2, sublinear_tf=True)
    vectors = vectorizer.fit_transform(texts)
    train_indices = [i for i, d in enumerate(docs) if d["split"] == "train"]
    held_indices = [i for i, d in enumerate(docs) if d["split"] != "train"]
    candidates = []
    for start in range(0, len(held_indices), 64):
        subset = held_indices[start:start+64]
        similarities = (vectors[subset] @ vectors[train_indices].T).toarray()
        best_positions = similarities.argmax(axis=1)
        for i, position, scores in zip(subset, best_positions, similarities):
            score = float(scores[position])
            if score >= .98:
                other = docs[train_indices[int(position)]]
                candidates.append(dict(document_id=docs[i]["document_id"], split=docs[i]["split"], label=docs[i]["label"],
                    nearest_train_id=other["document_id"], nearest_train_label=other["label"], cosine=score))
    write_csv(out / "near_text_candidates.csv", candidates, fields=["document_id", "split", "label", "nearest_train_id", "nearest_train_label", "cosine"])
    result["near_text_candidates_cosine_ge_098"] = dict(Counter(c["split"] for c in candidates))
    sources = rows(ROOT / "data/dataset_sources_extra.csv")
    source_by_id = {s["id"]: s for s in sources}
    source_groups = Counter((d["split"], d["label"], source_by_id.get(d["document_id"], {}).get("source_name", "unknown_original_1000")) for d in docs)
    write_csv(out / "source_by_split.csv", [dict(split=s, label=l, source=src, documents=n) for (s,l,src),n in sorted(source_groups.items())])
    pdf_metadata, errors = [], []
    for doc in docs:
        path = resolve(doc["raw_path"])
        if path.suffix.casefold() != ".pdf":
            continue
        try:
            with fitz.open(path) as pdf:
                rotated = [i for i, page in enumerate(pdf) if page.rotation]
                if rotated:
                    pdf_metadata.append(dict(id=doc["document_id"], split=doc["split"], rotated_indices=rotated, total_pages=len(pdf)))
        except Exception as exc:
            errors.append(dict(id=doc["document_id"], error_type=type(exc).__name__))
    result["raw_pdf_count"] = sum(resolve(d["raw_path"]).suffix.casefold() == ".pdf" for d in docs)
    result["pdf_documents_with_rotation"] = pdf_metadata
    result["pdf_metadata_errors"] = errors
    result["language_note"] = "Only source metadata hr is confirmed; no automatic language classifier used"
    dump(out / "summary.json", result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['texts_under_20_chars','pdf_documents_with_rotation']}, indent=2))


if __name__ == "__main__":
    main()
