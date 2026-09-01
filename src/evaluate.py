"""Print and validate the saved internal test results for all three models.

Training scripts already perform document-level test evaluation after training.
This entry point does not retrain or rerun inference; it provides one lightweight
check of the saved evaluation artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
REQUIRED_METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "seconds_per_document",
)
REQUIRED_ARTIFACTS = (
    "test_metrics.json",
    "classification_report.txt",
    "confusion_matrix.csv",
    "confusion_matrix.png",
)
MODEL_RESULTS = {
    "ResNet50": ("resnet50_multipage", "resnet50"),
    "XLM-RoBERTa": ("xlm_roberta_multipage", "xlm_roberta"),
    "LayoutLMv3": ("layoutlmv3_multipage", "layoutlmv3"),
}


def preferred_results_dir(multipage_name: str, legacy_name: str) -> Path:
    multipage_dir = RESULTS_ROOT / multipage_name
    if (multipage_dir / "test_metrics.json").is_file():
        return multipage_dir
    return RESULTS_ROOT / legacy_name


def load_model_result(model_name: str, result_dir: Path) -> dict[str, object]:
    missing_files = [name for name in REQUIRED_ARTIFACTS if not (result_dir / name).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing evaluation artifacts for {model_name} in {result_dir}: "
            + ", ".join(missing_files)
        )

    metrics_path = result_dir / "test_metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid metrics JSON: {metrics_path}") from error

    missing_metrics = [key for key in REQUIRED_METRICS if key not in metrics]
    if missing_metrics:
        raise KeyError(
            f"Missing metrics in {metrics_path}: {', '.join(missing_metrics)}"
        )
    return {
        "model": model_name,
        "result_dir": result_dir,
        **{key: float(metrics[key]) for key in REQUIRED_METRICS},
        "documents_evaluated": metrics.get("documents_evaluated", "n/a"),
    }


def print_results(rows: list[dict[str, object]]) -> None:
    print("INTERNAL DOCUMENT-LEVEL TEST RESULTS")
    print("-" * 104)
    print(
        f"{'Model':<14} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} "
        f"{'Macro F1':>10} {'Sec/doc':>10} {'Documents':>10}"
    )
    print("-" * 104)
    for row in rows:
        print(
            f"{str(row['model']):<14} "
            f"{float(row['accuracy']):>10.4f} "
            f"{float(row['macro_precision']):>10.4f} "
            f"{float(row['macro_recall']):>10.4f} "
            f"{float(row['macro_f1']):>10.4f} "
            f"{float(row['seconds_per_document']):>10.4f} "
            f"{str(row['documents_evaluated']):>10}"
        )
    print("-" * 104)
    print("Saved test artifacts are present and valid. No training or inference was run.")


def main() -> None:
    rows = [
        load_model_result(model_name, preferred_results_dir(*directory_names))
        for model_name, directory_names in MODEL_RESULTS.items()
    ]
    print_results(rows)


if __name__ == "__main__":
    main()
