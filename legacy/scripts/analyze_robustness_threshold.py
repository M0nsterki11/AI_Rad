import csv
import math
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "results" / "external_robus_test" / "all_predictions.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "robustness_test"
DETAILED_CSV_PATH = OUTPUT_DIR / "all_predictions_detailed.csv"
REPORT_PATH = OUTPUT_DIR / "threshold_analysis.txt"

THRESHOLD = 0.50
HIGH_CONFIDENCE_THRESHOLD = 0.90
EXPECTED_DOCUMENTS_PER_MODEL = 12
CLASS_NAMES = ["invoice", "cv", "contract", "email", "scientific"]
MODEL_ORDER = ["resnet50", "xlm_roberta", "layoutlmv3"]
MODEL_NAMES = {
    "resnet50": "ResNet50",
    "xlm_roberta": "XLM-RoBERTa",
    "layoutlmv3": "LayoutLMv3",
}
OUTPUT_FIELDS = [
    "model",
    "document",
    "true_label",
    "predicted_label",
    "confidence",
    "top1_correct",
    "above_50_threshold",
    "accepted_by_50_threshold",
    "technical_status",
    "technical_error",
]


def read_source_rows():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Nedostaje postojeći robustness rezultat: {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {
            "model",
            "document_path",
            "true_label",
            "predicted_label",
            "confidence",
            "status",
            "error_message",
            *[f"prob_{label}" for label in CLASS_NAMES],
        }
        missing_columns = sorted(required_columns - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(
                "Ulazni CSV nema potrebne stupce: " + ", ".join(missing_columns)
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"Ulazni CSV je prazan: {INPUT_PATH}")
    return rows


def parse_probability(value, row_description):
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Nevaljana vjerojatnost za {row_description}: {value!r}") from error


def validate_and_transform(rows):
    counts = Counter(row["model"] for row in rows)
    unknown_models = sorted(set(counts) - set(MODEL_ORDER))
    if unknown_models:
        raise ValueError("Nepoznati modeli u ulaznom CSV-u: " + ", ".join(unknown_models))

    for model_key in MODEL_ORDER:
        if counts[model_key] != EXPECTED_DOCUMENTS_PER_MODEL:
            raise ValueError(
                f"{MODEL_NAMES[model_key]} ima {counts[model_key]} redova; "
                f"očekivano je {EXPECTED_DOCUMENTS_PER_MODEL}."
            )

    document_sets = {
        model_key: {
            row["document_path"] for row in rows if row["model"] == model_key
        }
        for model_key in MODEL_ORDER
    }
    reference_documents = document_sets[MODEL_ORDER[0]]
    for model_key in MODEL_ORDER[1:]:
        if document_sets[model_key] != reference_documents:
            raise ValueError(
                f"{MODEL_NAMES[model_key]} nema isti skup dokumenata kao ResNet50."
            )

    transformed = []
    seen_pairs = set()
    for row in rows:
        model_key = row["model"]
        document = row["document_path"]
        pair = (model_key, document)
        if pair in seen_pairs:
            raise ValueError(f"Dupli model/dokument red: {model_key}, {document}")
        seen_pairs.add(pair)

        status = row["status"].strip().lower()
        if status not in {"success", "failed"}:
            raise ValueError(f"Nepoznat technical status {status!r} za {document}")

        predicted_label = row["predicted_label"].strip()
        confidence = None
        top1_correct = False
        above_threshold = False

        if status == "success":
            if predicted_label not in CLASS_NAMES:
                raise ValueError(
                    f"Nevaljana Top-1 klasa {predicted_label!r} za {model_key}, {document}"
                )

            probabilities = {
                label: parse_probability(
                    row[f"prob_{label}"],
                    f"{model_key}, {document}, prob_{label}",
                )
                for label in CLASS_NAMES
            }
            top1_label = max(CLASS_NAMES, key=lambda label: probabilities[label])
            top1_probability = probabilities[top1_label]
            confidence = parse_probability(
                row["confidence"],
                f"{model_key}, {document}, confidence",
            )

            if predicted_label != top1_label:
                raise ValueError(
                    f"Spremljeni predicted_label nije Top-1 za {model_key}, {document}: "
                    f"spremljeno={predicted_label}, izračunato={top1_label}"
                )
            if not math.isclose(confidence, top1_probability, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(
                    f"Confidence nije Top-1 vjerojatnost za {model_key}, {document}: "
                    f"spremljeno={confidence}, izračunato={top1_probability}"
                )

            top1_correct = predicted_label == row["true_label"]
            above_threshold = confidence >= THRESHOLD
        elif predicted_label or row["confidence"].strip():
            raise ValueError(
                f"Tehnički neuspješan red neočekivano sadrži predikciju: "
                f"{model_key}, {document}"
            )

        transformed.append(
            {
                "model": MODEL_NAMES[model_key],
                "document": document,
                "true_label": row["true_label"],
                "predicted_label": predicted_label,
                "confidence": "" if confidence is None else f"{confidence:.12f}",
                "top1_correct": top1_correct,
                "above_50_threshold": above_threshold,
                "accepted_by_50_threshold": status == "success" and above_threshold,
                "technical_status": status,
                "technical_error": row["error_message"].strip(),
            }
        )

    model_rank = {MODEL_NAMES[key]: index for index, key in enumerate(MODEL_ORDER)}
    transformed.sort(key=lambda row: (model_rank[row["model"]], row["document"]))
    return transformed


def write_detailed_csv(rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with DETAILED_CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def percent(numerator, denominator):
    if denominator == 0:
        return "N/A"
    return f"{100 * numerator / denominator:.2f}%"


def analyze_model(rows, model_name):
    model_rows = [row for row in rows if row["model"] == model_name]
    successful = [row for row in model_rows if row["technical_status"] == "success"]
    failed = [row for row in model_rows if row["technical_status"] == "failed"]
    correct = [row for row in successful if row["top1_correct"]]
    correct_below = [
        row
        for row in successful
        if row["top1_correct"] and not row["above_50_threshold"]
    ]
    wrong_above_50 = [
        row
        for row in successful
        if not row["top1_correct"] and row["above_50_threshold"]
    ]
    wrong_above_90 = [
        row
        for row in successful
        if not row["top1_correct"] and float(row["confidence"]) >= HIGH_CONFIDENCE_THRESHOLD
    ]
    accepted = [row for row in model_rows if row["accepted_by_50_threshold"]]
    accepted_correct = [row for row in accepted if row["top1_correct"]]
    threshold_rejected = [
        row
        for row in successful
        if not row["above_50_threshold"]
    ]

    return {
        "rows": model_rows,
        "total": len(model_rows),
        "successful": successful,
        "failed": failed,
        "correct": correct,
        "correct_below": correct_below,
        "wrong_above_50": wrong_above_50,
        "wrong_above_90": wrong_above_90,
        "accepted": accepted,
        "accepted_correct": accepted_correct,
        "threshold_rejected": threshold_rejected,
    }


def format_case(row):
    confidence = float(row["confidence"])
    return (
        f"- {row['document']} | true={row['true_label']} | "
        f"Top-1={row['predicted_label']} | confidence={confidence:.2%}"
    )


def build_report(rows):
    lines = [
        "ROBUSTNESS TEST - THRESHOLD ANALYSIS",
        "=" * 45,
        "",
        f"Izvor: {INPUT_PATH.relative_to(PROJECT_ROOT)}",
        f"Aplikacijski threshold: confidence >= {THRESHOLD:.2f}",
        "",
        "VAŽNO:",
        "- technical_status=failed označava samo stvarnu preprocessing/OCR/model grešku.",
        "- Confidence ispod 50% ne mijenja technical_status; takav red ostaje success.",
        "- Top-1 accuracy nad svih 12 dokumenata računa tehničke failove kao netočne.",
        "- Accuracy prihvaćenih predikcija računa se samo nad success redovima s confidence >= 50%.",
        "",
    ]

    analyses = {}
    for model_key in MODEL_ORDER:
        model_name = MODEL_NAMES[model_key]
        analysis = analyze_model(rows, model_name)
        analyses[model_name] = analysis

        lines.extend(
            [
                model_name,
                "-" * len(model_name),
                f"1. Ukupan broj dokumenata: {analysis['total']}",
                f"2. Tehnički uspješno obrađeni: {len(analysis['successful'])}",
                f"3. Stvarni tehnički failovi: {len(analysis['failed'])}",
                f"4. Top-1 točne predikcije bez thresholda: "
                f"{len(analysis['correct'])}/{analysis['total']}",
                f"5. Top-1 accuracy bez thresholda (svih dokumenata): "
                f"{percent(len(analysis['correct']), analysis['total'])}",
                f"   Top-1 accuracy među tehnički uspješnima: "
                f"{percent(len(analysis['correct']), len(analysis['successful']))}",
                f"6. Točne Top-1 predikcije s confidence < 50%: "
                f"{len(analysis['correct_below'])}",
                f"7. Pogrešne Top-1 predikcije s confidence >= 50%: "
                f"{len(analysis['wrong_above_50'])}",
                f"8. Accuracy samo prihvaćenih predikcija >= 50%: "
                f"{percent(len(analysis['accepted_correct']), len(analysis['accepted']))} "
                f"({len(analysis['accepted_correct'])}/{len(analysis['accepted'])})",
                f"9. Tehnički uspješni dokumenti odbačeni samo thresholdom: "
                f"{len(analysis['threshold_rejected'])}",
                f"   Ukupno neprihvaćeni (threshold + tehnički failovi): "
                f"{analysis['total'] - len(analysis['accepted'])}",
                "10. Točne Top-1 predikcije ispod 50%:",
            ]
        )
        if analysis["correct_below"]:
            lines.extend(format_case(row) for row in analysis["correct_below"])
        else:
            lines.append("- Nema takvih slučajeva.")

        lines.append("")
        lines.append("Pogrešne predikcije s confidence >= 90%:")
        if analysis["wrong_above_90"]:
            lines.extend(format_case(row) for row in analysis["wrong_above_90"])
        else:
            lines.append("- Nema takvih slučajeva.")

        lines.append("")
        lines.append("Stvarni tehnički failovi:")
        if analysis["failed"]:
            for row in analysis["failed"]:
                lines.append(
                    f"- {row['document']} | error={row['technical_error']}"
                )
        else:
            lines.append("- Nema tehničkih failova.")
        lines.extend(["", ""])

    return "\n".join(lines).rstrip() + "\n", analyses


def print_terminal_summary(analyses):
    for model_key in MODEL_ORDER:
        model_name = MODEL_NAMES[model_key]
        analysis = analyses[model_name]
        print(model_name)
        print(f"Technical failures: {len(analysis['failed'])}/{analysis['total']}")
        print(f"Top-1 correct: {len(analysis['correct'])}/{analysis['total']}")
        print(f"Correct but below 50% threshold: {len(analysis['correct_below'])}")
        print(f"Wrong but >=50% confidence: {len(analysis['wrong_above_50'])}")
        print(f"Wrong but >=90% confidence: {len(analysis['wrong_above_90'])}")
        print()

    print(f"Detailed CSV: {DETAILED_CSV_PATH}")
    print(f"Threshold report: {REPORT_PATH}")


def main():
    source_rows = read_source_rows()
    detailed_rows = validate_and_transform(source_rows)
    write_detailed_csv(detailed_rows)
    report, analyses = build_report(detailed_rows)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print_terminal_summary(analyses)


if __name__ == "__main__":
    main()
