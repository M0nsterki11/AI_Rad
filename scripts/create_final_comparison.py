import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
EXTERNAL_COMPARISON_PATH = RESULTS_DIR / "external_test" / "comparison_metrics.csv"
FINAL_CSV_PATH = RESULTS_DIR / "final_comparison.csv"
FINAL_MD_PATH = RESULTS_DIR / "final_comparison.md"
FINAL_PNG_PATH = RESULTS_DIR / "final_comparison.png"
FINAL_SUMMARY_PATH = RESULTS_DIR / "final_comparison_summary.txt"


def preferred_results_dir(multipage_name, legacy_name):
    multipage_dir = RESULTS_DIR / multipage_name
    if (multipage_dir / "test_metrics.json").is_file():
        return multipage_dir
    return RESULTS_DIR / legacy_name

MODELS = [
    {
        "name": "ResNet50",
        "internal_dir": preferred_results_dir("resnet50_multipage", "resnet50"),
        "external_name": "ResNet50",
    },
    {
        "name": "XLM-RoBERTa",
        "internal_dir": preferred_results_dir("xlm_roberta_multipage", "xlm_roberta"),
        "external_name": "XLM-RoBERTa",
    },
    {
        "name": "LayoutLMv3",
        "internal_dir": preferred_results_dir("layoutlmv3_multipage", "layoutlmv3"),
        "external_name": "LayoutLMv3",
    },
]

METRIC_KEYS = [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "seconds_per_document",
]
REQUIRED_SCORE_KEYS = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
TIME_KEYS = ["seconds_per_document", "seconds_per_sample"]


def safe_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def require_float(source, key, value):
    if value in (None, ""):
        raise KeyError(f"Missing required metric '{key}' in {source}")
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Metric '{key}' in {source} is not a valid number: {value!r}") from error


def find_internal_metrics_file(results_dir):
    preferred = results_dir / "test_metrics.json"
    if preferred.exists():
        return preferred

    candidates = sorted(results_dir.glob("*metrics*.json"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Missing internal metrics JSON in {results_dir}. Expected test_metrics.json or *metrics*.json."
    )


def load_json_metrics(results_dir):
    path = find_internal_metrics_file(results_dir)

    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = {
        key: require_float(path, key, data.get(key))
        for key in REQUIRED_SCORE_KEYS
    }

    time_key = next((key for key in TIME_KEYS if key in data), None)
    if time_key is None:
        raise KeyError(
            f"Missing required timing metric in {path}. Expected one of: {', '.join(TIME_KEYS)}"
        )
    metrics["seconds_per_document"] = require_float(path, time_key, data.get(time_key))
    return metrics


def load_external_metrics(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing external comparison file: {path}")

    rows = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        required_columns = {"model", *METRIC_KEYS}
        missing_columns = required_columns.difference(fieldnames)
        if missing_columns:
            raise KeyError(
                f"Missing columns in {path}: {', '.join(sorted(missing_columns))}"
            )

        for row in reader:
            model_name = str(row.get("model", "")).strip()
            if not model_name:
                raise ValueError(f"Found external metrics row without model name in {path}")
            rows[model_name] = {
                key: require_float(path, key, row.get(key))
                for key in METRIC_KEYS
            }
    return rows


def build_final_rows():
    external_metrics = load_external_metrics(EXTERNAL_COMPARISON_PATH)
    rows = []

    for model in MODELS:
        internal = load_json_metrics(model["internal_dir"])
        external = external_metrics.get(model["external_name"])
        if external is None:
            raise ValueError(f"Missing external metrics for model: {model['external_name']}")

        row = {
            "model": model["name"],
            "internal_accuracy": internal["accuracy"],
            "external_accuracy": external["accuracy"],
            "accuracy_drop": internal["accuracy"] - external["accuracy"],
            "internal_macro_precision": internal["macro_precision"],
            "external_macro_precision": external["macro_precision"],
            "internal_macro_recall": internal["macro_recall"],
            "external_macro_recall": external["macro_recall"],
            "internal_macro_f1": internal["macro_f1"],
            "external_macro_f1": external["macro_f1"],
            "macro_f1_drop": internal["macro_f1"] - external["macro_f1"],
            "internal_seconds_per_document": internal["seconds_per_document"],
            "external_seconds_per_document": external["seconds_per_document"],
        }
        rows.append(row)

    return rows


def write_csv(rows):
    FINAL_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "internal_accuracy",
        "external_accuracy",
        "accuracy_drop",
        "internal_macro_precision",
        "external_macro_precision",
        "internal_macro_recall",
        "external_macro_recall",
        "internal_macro_f1",
        "external_macro_f1",
        "macro_f1_drop",
        "internal_seconds_per_document",
        "external_seconds_per_document",
    ]
    with FINAL_CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percent(value):
    return f"{value * 100:.2f}%"


def seconds(value):
    return f"{value:.4f}s"


def write_markdown(rows):
    lines = [
        "# Interni test vs. vanjski test",
        "",
        "| Model | Interni accuracy | Vanjski accuracy | Pad accuracy | Interni macro precision | Vanjski macro precision | Interni macro recall | Vanjski macro recall | Interni macro F1 | Vanjski macro F1 | Pad macro F1 | Interni sec/doc | Vanjski sec/doc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['model']} | "
            f"{percent(row['internal_accuracy'])} | "
            f"{percent(row['external_accuracy'])} | "
            f"{percent(row['accuracy_drop'])} | "
            f"{percent(row['internal_macro_precision'])} | "
            f"{percent(row['external_macro_precision'])} | "
            f"{percent(row['internal_macro_recall'])} | "
            f"{percent(row['external_macro_recall'])} | "
            f"{percent(row['internal_macro_f1'])} | "
            f"{percent(row['external_macro_f1'])} | "
            f"{percent(row['macro_f1_drop'])} | "
            f"{seconds(row['internal_seconds_per_document'])} | "
            f"{seconds(row['external_seconds_per_document'])} |"
        )
    FINAL_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_fonts():
    try:
        return (
            ImageFont.truetype("arial.ttf", 24),
            ImageFont.truetype("arial.ttf", 16),
            ImageFont.truetype("arial.ttf", 13),
        )
    except OSError:
        return (ImageFont.load_default(), ImageFont.load_default(), ImageFont.load_default())


def draw_final_chart(rows):
    width = 1180
    height = 620
    left = 90
    right = 40
    top = 90
    bottom = 95
    plot_width = width - left - right
    plot_height = height - top - bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, small_font = load_fonts()

    colors = {
        "internal_accuracy": (52, 112, 184),
        "external_accuracy": (82, 163, 100),
        "internal_macro_f1": (230, 146, 52),
        "external_macro_f1": (164, 86, 176),
    }
    series = [
        ("internal_accuracy", "Internal accuracy"),
        ("external_accuracy", "External accuracy"),
        ("internal_macro_f1", "Internal macro F1"),
        ("external_macro_f1", "External macro F1"),
    ]

    draw.text((left, 30), "Internal vs External Model Metrics", font=title_font, fill="black")

    for tick in range(0, 6):
        value = tick / 5
        y = top + plot_height - int(value * plot_height)
        draw.line((left, y, width - right, y), fill=(225, 225, 225))
        draw.text((20, y - 8), f"{value * 100:.0f}%", font=small_font, fill=(70, 70, 70))

    group_width = plot_width / len(rows)
    bar_width = 34
    gap = 8

    for group_index, row in enumerate(rows):
        group_left = left + group_index * group_width
        center = group_left + group_width / 2
        total_bar_width = len(series) * bar_width + (len(series) - 1) * gap
        start_x = center - total_bar_width / 2

        for series_index, (key, _) in enumerate(series):
            value = max(0.0, min(1.0, row[key]))
            x1 = int(start_x + series_index * (bar_width + gap))
            x2 = x1 + bar_width
            y1 = top + plot_height - int(value * plot_height)
            y2 = top + plot_height
            draw.rectangle((x1, y1, x2, y2), fill=colors[key])
            draw.text((x1 - 4, y1 - 18), f"{value * 100:.0f}", font=small_font, fill=(45, 45, 45))

        label_bbox = draw.textbbox((0, 0), row["model"], font=label_font)
        label_width = label_bbox[2] - label_bbox[0]
        draw.text((center - label_width / 2, top + plot_height + 18), row["model"], font=label_font, fill="black")

    legend_x = left
    legend_y = height - 45
    for key, label in series:
        draw.rectangle((legend_x, legend_y, legend_x + 18, legend_y + 18), fill=colors[key])
        draw.text((legend_x + 25, legend_y), label, font=small_font, fill="black")
        legend_x += 220

    draw.line((left, top, left, top + plot_height), fill=(90, 90, 90), width=2)
    draw.line((left, top + plot_height, width - right, top + plot_height), fill=(90, 90, 90), width=2)
    image.save(FINAL_PNG_PATH)


def build_summary(rows):
    best_internal = max(rows, key=lambda row: row["internal_macro_f1"])
    best_external = max(rows, key=lambda row: row["external_macro_f1"])
    smallest_drop = min(rows, key=lambda row: row["macro_f1_drop"])

    return "\n".join(
        [
            f"Na internom testnom skupu najbolji model je {best_internal['model']} "
            f"(macro F1: {percent(best_internal['internal_macro_f1'])}).",
            f"Na vanjskom testnom skupu najbolji model je {best_external['model']} "
            f"(macro F1: {percent(best_external['external_macro_f1'])}).",
            f"Najmanji pad macro F1 rezultata ima {smallest_drop['model']} "
            f"(pad: {percent(smallest_drop['macro_f1_drop'])}).",
            "Razlika izmedu internih i vanjskih rezultata pokazuje koliko modeli "
            "generaliziraju na dokumente iz drugih izvora.",
        ]
    )


def write_summary(rows):
    FINAL_SUMMARY_PATH.write_text(build_summary(rows) + "\n", encoding="utf-8")


def print_table(rows):
    print()
    print("FINAL COMPARISON")
    print("-" * 132)
    print(
        f"{'Model':<14} {'Int Acc':>9} {'Ext Acc':>9} {'Acc Drop':>9} "
        f"{'Int F1':>9} {'Ext F1':>9} {'F1 Drop':>9} {'Int s/doc':>10} {'Ext s/doc':>10}"
    )
    print("-" * 132)
    for row in rows:
        print(
            f"{row['model']:<14} "
            f"{percent(row['internal_accuracy']):>9} "
            f"{percent(row['external_accuracy']):>9} "
            f"{percent(row['accuracy_drop']):>9} "
            f"{percent(row['internal_macro_f1']):>9} "
            f"{percent(row['external_macro_f1']):>9} "
            f"{percent(row['macro_f1_drop']):>9} "
            f"{seconds(row['internal_seconds_per_document']):>10} "
            f"{seconds(row['external_seconds_per_document']):>10}"
        )
    print("-" * 132)
    print(f"CSV: {FINAL_CSV_PATH}")
    print(f"Markdown: {FINAL_MD_PATH}")
    print(f"Chart: {FINAL_PNG_PATH}")
    print(f"Summary: {FINAL_SUMMARY_PATH}")
    print()
    print(build_summary(rows))


def main():
    rows = build_final_rows()
    write_csv(rows)
    write_markdown(rows)
    draw_final_chart(rows)
    write_summary(rows)
    print_table(rows)


if __name__ == "__main__":
    main()
