"""Combine retained before/after artifacts; never alter their measurements."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_models import MODELS, LABELS, rows, dump, write_csv


def main():
    before = ROOT / "reports/baseline_20260915"
    after = ROOT / "reports/after_20260915"
    differences, class_rows, summary, errors = [], [], {}, []
    for model in MODELS:
        old = rows(before / model / "predictions.csv")
        new = rows(after / model / "predictions.csv")
        old_map = {(r["document_id"], r["pathway"]): r for r in old}
        for r in new:
            if r["pathway"] != "app_fresh":
                continue
            a = old_map[(r["document_id"], r["pathway"])]
            differences.append(dict(model=model, dataset=r["dataset"], document_id=r["document_id"], true_label=r["true_label"],
                before_prediction=a["predicted_label"], after_prediction=r["predicted_label"],
                before_confidence=a["confidence"], after_confidence=r["confidence"],
                prediction_changed=a["predicted_label"] != r["predicted_label"],
                confidence_abs_difference=abs(float(a["confidence"])-float(r["confidence"])) if a["confidence"] and r["confidence"] else None,
                before_error=a["error"], after_error=r["error"]))
            if r["error"] or r["predicted_label"] != r["true_label"]:
                errors.append(r)
        current_metrics = json.loads((after / model / "metrics.json").read_text(encoding="utf-8"))
        summary[model] = {"metrics": current_metrics, "input_comparisons": {}}
        comparisons = json.loads((before / "corrected_harness" / model / "input_comparisons.json").read_text(encoding="utf-8"))
        for pair in ["evaluation_vs_app_stored", "app_stored_vs_fresh"]:
            comparable = [d[pair] for d in comparisons if d[pair]["items_left"] > 0 and d[pair]["items_right"] > 0]
            same = [d for d in comparable if d["canonical_inputs_equal"]]
            summary[model]["input_comparisons"][pair] = dict(documents=100, comparable=len(comparable),
                identical_input_documents=len(same), max_logit_delta_on_identical_inputs=max((d["max_abs_logit_difference"] or 0 for d in same), default=None),
                max_logit_delta_any_comparable=max((d["max_abs_logit_difference"] or 0 for d in comparable), default=None))
        for dataset in ["internal_test", "external_existing"]:
            metric = current_metrics[dataset+"/app_fresh"]
            for label in LABELS:
                class_rows.append(dict(model=model, dataset=dataset, label=label, **metric["per_class"][label]))
        old_runtime = json.loads((before / model / "runtime_model.json").read_text(encoding="utf-8"))
        new_runtime = json.loads((after / model / "runtime_model.json").read_text(encoding="utf-8"))
        summary[model]["weight_sha256_unchanged"] = old_runtime["weight_sha256"] == new_runtime["weight_sha256"]
        subset = [d for d in differences if d["model"] == model]
        summary[model]["before_after"] = dict(documents=len(subset), changed_predictions=sum(d["prediction_changed"] for d in subset),
            max_confidence_delta=max((d["confidence_abs_difference"] or 0) for d in subset),
            errors_before=sum(bool(d["before_error"]) for d in subset), errors_after=sum(bool(d["after_error"]) for d in subset))
    write_csv(ROOT / "reports/before_after_predictions.csv", differences)
    write_csv(ROOT / "reports/per_class_metrics.csv", class_rows)
    write_csv(ROOT / "reports/errors_and_misclassifications.csv", errors)
    dump(ROOT / "reports/metrics_summary.json", summary)
    print(json.dumps({m:{"before_after":v["before_after"],"inputs":v["input_comparisons"]} for m,v in summary.items()},indent=2))


if __name__ == "__main__":
    main()
