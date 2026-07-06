import argparse
import csv
import inspect
import json
import random
import time
from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError as error:
    raise SystemExit(
        "Missing required library torch. Install project requirements before running this script."
    ) from error

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as error:
    raise SystemExit(
        "Missing required library Pillow. Install it with: python -m pip install pillow"
    ) from error

try:
    from transformers import AutoModelForSequenceClassification, AutoProcessor
except ImportError as error:
    raise SystemExit(
        "Missing required library transformers. Install it with: python -m pip install transformers"
    ) from error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
MODEL_DIR = PROJECT_ROOT / "models" / "layoutlmv3"
RESULTS_DIR = PROJECT_ROOT / "results" / "layoutlmv3"
MODEL_NAME = "microsoft/layoutlmv3-base"
RANDOM_SEED = 42

DEFAULT_CLASS_NAMES = ["invoice", "cv", "contract", "email", "scientific"]
RESNET_LABEL_MAPPING_PATH = PROJECT_ROOT / "models" / "resnet50" / "label_mapping.json"
LABEL_MAPPING_PATH = MODEL_DIR / "label_mapping.json"
MODEL_INPUT_KEYS = {
    "input_ids",
    "attention_mask",
    "bbox",
    "pixel_values",
    "token_type_ids",
    "labels",
}


class LayoutLMv3DocumentDataset(Dataset):
    def __init__(self, rows, processor, max_length, label_to_index):
        self.rows = rows
        self.processor = processor
        self.max_length = max_length
        self.label_to_index = label_to_index

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        row_id = row["id"]
        image_path = resolve_project_path(row["image_path"])
        ocr_path = resolve_project_path(row["ocr_path"])

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Missing image for {row_id}: {image_path}") from error
        except Exception as error:
            raise RuntimeError(f"Could not open image for {row_id}: {image_path}") from error

        words, boxes = load_and_normalize_ocr(row_id, ocr_path, image)

        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {}
        for key, value in encoding.items():
            if torch.is_tensor(value):
                item[key] = value.squeeze(0)
            else:
                item[key] = value

        item["labels"] = torch.tensor(self.label_to_index[row["label"]], dtype=torch.long)
        item["id"] = row_id
        item["image_path"] = row["image_path"]
        item["ocr_path"] = row["ocr_path"]
        return item


def parse_args():
    parser = argparse.ArgumentParser(description="Train LayoutLMv3 document classifier.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_project_path(value):
    path = Path(str(value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def save_label_mapping(path, class_names, label_to_index):
    index_to_label = {str(index): label for label, index in label_to_index.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "class_names": class_names,
                "label_to_index": label_to_index,
                "index_to_label": index_to_label,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_label_mapping():
    if RESNET_LABEL_MAPPING_PATH.exists():
        mapping = json.loads(RESNET_LABEL_MAPPING_PATH.read_text(encoding="utf-8"))
        class_names = mapping.get("class_names") or DEFAULT_CLASS_NAMES
        label_to_index = mapping.get("label_to_index") or {
            label: index for index, label in enumerate(class_names)
        }
    else:
        class_names = DEFAULT_CLASS_NAMES
        label_to_index = {label: index for index, label in enumerate(class_names)}

    if class_names != DEFAULT_CLASS_NAMES:
        raise ValueError(f"Unexpected class order from label mapping: {class_names}")

    expected = {label: index for index, label in enumerate(DEFAULT_CLASS_NAMES)}
    if label_to_index != expected:
        raise ValueError(f"Unexpected label_to_index mapping: {label_to_index}")

    save_label_mapping(LABEL_MAPPING_PATH, class_names, label_to_index)
    return class_names, label_to_index


def read_split(split_name):
    path = SPLITS_DIR / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split CSV file: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])

    required = {"id", "label", "image_path", "ocr_path"}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")

    return rows


def validate_and_prepare_splits(label_to_index):
    rows_by_split = {}

    for split_name in ("train", "validation", "test"):
        rows = read_split(split_name)
        prepared_rows = []
        missing_images = []
        missing_ocr = []
        invalid_labels = []

        for row in rows:
            row_id = str(row.get("id", "")).strip()
            label = str(row.get("label", "")).strip()
            image_path = str(row.get("image_path", "")).strip()
            ocr_path = str(row.get("ocr_path", "")).strip()

            if label not in label_to_index:
                invalid_labels.append((row_id, label))
                continue

            if not image_path or not resolve_project_path(image_path).exists():
                missing_images.append((row_id, image_path or "(empty image_path)"))
                continue

            if not ocr_path or not resolve_project_path(ocr_path).exists():
                missing_ocr.append((row_id, ocr_path or "(empty ocr_path)"))
                continue

            row["id"] = row_id
            row["label"] = label
            row["image_path"] = image_path
            row["ocr_path"] = ocr_path
            prepared_rows.append(row)

        if invalid_labels:
            examples = "\n".join(f"  {row_id}: {label}" for row_id, label in invalid_labels[:20])
            raise ValueError(f"{split_name}.csv has invalid labels:\n{examples}")
        if missing_images:
            examples = "\n".join(f"  {row_id}: {path}" for row_id, path in missing_images[:20])
            raise FileNotFoundError(f"{split_name}.csv has missing images:\n{examples}")
        if missing_ocr:
            examples = "\n".join(f"  {row_id}: {path}" for row_id, path in missing_ocr[:20])
            raise FileNotFoundError(f"{split_name}.csv has missing OCR JSON files:\n{examples}")

        rows_by_split[split_name] = prepared_rows
        counts = {label: 0 for label in DEFAULT_CLASS_NAMES}
        for row in prepared_rows:
            counts[row["label"]] += 1
        print(f"{split_name}: {len(prepared_rows)} rows | {counts}")

    return rows_by_split


def limit_for_smoke_test(rows_by_split, max_per_class=5):
    rng = random.Random(RANDOM_SEED)
    limited = {}

    for split_name, rows in rows_by_split.items():
        split_rows = []
        for label in DEFAULT_CLASS_NAMES:
            label_rows = [row for row in rows if row["label"] == label]
            rng.shuffle(label_rows)
            split_rows.extend(label_rows[:max_per_class])
        rng.shuffle(split_rows)
        limited[split_name] = split_rows

    return limited


def get_image_dimension(ocr_data, key, fallback):
    value = ocr_data.get(key, fallback)
    try:
        value = int(float(value))
    except (TypeError, ValueError):
        value = int(fallback)
    return value if value > 0 else int(fallback)


def clamp_0_1000(value):
    return max(0, min(1000, int(round(value))))


def normalize_box(row_id, raw_box, image_width, image_height):
    if (
        not isinstance(raw_box, list)
        or len(raw_box) != 4
        or not all(isinstance(value, (int, float)) for value in raw_box)
    ):
        raise ValueError(f"Invalid box shape for {row_id}: {raw_box}")

    x1, y1, x2, y2 = [float(value) for value in raw_box]
    if x2 < x1 or y2 < y1:
        raise ValueError(f"Invalid raw box coordinates for {row_id}: {raw_box}")

    normalized = [
        clamp_0_1000(1000 * x1 / image_width),
        clamp_0_1000(1000 * y1 / image_height),
        clamp_0_1000(1000 * x2 / image_width),
        clamp_0_1000(1000 * y2 / image_height),
    ]

    nx1, ny1, nx2, ny2 = normalized
    if not (0 <= nx1 <= nx2 <= 1000 and 0 <= ny1 <= ny2 <= 1000):
        raise ValueError(f"Invalid normalized box for {row_id}: {normalized}")

    return normalized


def load_and_normalize_ocr(row_id, ocr_path, image):
    if not ocr_path.exists():
        raise FileNotFoundError(f"Missing OCR JSON for {row_id}: {ocr_path}")

    try:
        with ocr_path.open("r", encoding="utf-8") as file:
            ocr_data = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Could not parse OCR JSON for {row_id}: {ocr_path}") from error

    words = ocr_data.get("words")
    boxes = ocr_data.get("boxes")

    if not isinstance(words, list) or not isinstance(boxes, list):
        raise ValueError(f"OCR JSON for {row_id} must contain list fields 'words' and 'boxes'.")
    if len(words) != len(boxes):
        raise ValueError(
            f"OCR words/boxes length mismatch for {row_id}: {len(words)} words, {len(boxes)} boxes"
        )
    if not words:
        raise ValueError(f"OCR word list is empty for {row_id}: {ocr_path}")

    image_width = get_image_dimension(ocr_data, "image_width", image.width)
    image_height = get_image_dimension(ocr_data, "image_height", image.height)

    clean_words = []
    normalized_boxes = []
    for word, box in zip(words, boxes):
        text = str(word).strip()
        if not text:
            continue
        clean_words.append(text)
        normalized_boxes.append(normalize_box(row_id, box, image_width, image_height))

    if not clean_words:
        raise ValueError(f"OCR word list is empty after cleaning for {row_id}: {ocr_path}")
    if len(clean_words) != len(normalized_boxes):
        raise ValueError(f"Cleaned OCR words/boxes mismatch for {row_id}: {ocr_path}")

    return clean_words, normalized_boxes


def make_loaders(rows_by_split, processor, batch_size, max_length, label_to_index, num_workers):
    pin_memory = torch.cuda.is_available()
    return {
        split_name: DataLoader(
            LayoutLMv3DocumentDataset(rows, processor, max_length, label_to_index),
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for split_name, rows in rows_by_split.items()
    }


def model_input_keys(model):
    forward_keys = set(inspect.signature(model.forward).parameters)
    return MODEL_INPUT_KEYS.intersection(forward_keys.union({"labels"}))


def batch_to_model_inputs(batch, device, allowed_keys):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key in allowed_keys and torch.is_tensor(value)
    }


def move_labels_to_device(batch, device):
    return batch["labels"].to(device, non_blocking=True)


def classification_metrics(y_true, y_pred, class_names):
    per_class = {}
    precisions = []
    recalls = []
    f1_scores = []

    for index, label in enumerate(class_names):
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == index and pred == index)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != index and pred == index)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == index and pred != index)
        support = sum(1 for true in y_true if true == index)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true) if y_true else 0.0
    return {
        "accuracy": accuracy,
        "macro_precision": sum(precisions) / len(precisions),
        "macro_recall": sum(recalls) / len(recalls),
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "per_class": per_class,
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    allowed_input_keys,
    gradient_accumulation_steps,
    use_amp,
    class_names,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total = 0
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader, start=1):
        labels = move_labels_to_device(batch, device)
        model_inputs = batch_to_model_inputs(batch, device, allowed_input_keys)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(**model_inputs)
            loss = outputs.loss
            loss_for_backward = loss / gradient_accumulation_steps

        scaler.scale(loss_for_backward).backward()

        should_step = step % gradient_accumulation_steps == 0 or step == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        predictions = outputs.logits.detach().argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.detach().item() * batch_size
        total += batch_size
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())

    metrics = classification_metrics(y_true, y_pred, class_names)
    metrics["loss"] = total_loss / total if total else 0.0
    return metrics


def evaluate(
    model,
    loader,
    device,
    allowed_input_keys,
    class_names,
    use_amp,
    measure_prediction_time=False,
):
    model.eval()
    total_loss = 0.0
    total = 0
    y_true = []
    y_pred = []
    confidences = []
    ids = []
    image_paths = []
    ocr_paths = []
    prediction_time = 0.0

    with torch.no_grad():
        for batch in loader:
            labels = move_labels_to_device(batch, device)
            model_inputs = batch_to_model_inputs(batch, device, allowed_input_keys)

            if measure_prediction_time and device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**model_inputs)
            if measure_prediction_time and device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            if measure_prediction_time:
                prediction_time += end - start

            probabilities = torch.softmax(outputs.logits, dim=1)
            confidence_values, predictions = probabilities.max(dim=1)
            batch_size = labels.size(0)

            total_loss += outputs.loss.detach().item() * batch_size
            total += batch_size
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
            confidences.extend(confidence_values.cpu().tolist())
            ids.extend(batch["id"])
            image_paths.extend(batch["image_path"])
            ocr_paths.extend(batch["ocr_path"])

    metrics = classification_metrics(y_true, y_pred, class_names)
    metrics["loss"] = total_loss / total if total else 0.0
    metrics["prediction_time_seconds"] = prediction_time
    metrics["seconds_per_document"] = prediction_time / total if total else 0.0
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["confidences"] = confidences
    metrics["ids"] = ids
    metrics["image_paths"] = image_paths
    metrics["ocr_paths"] = ocr_paths
    return metrics


def classification_report_text(metrics, class_names):
    lines = ["label,precision,recall,f1,support"]
    for label in class_names:
        row = metrics["per_class"][label]
        lines.append(
            f"{label},{row['precision']:.6f},{row['recall']:.6f},{row['f1']:.6f},{row['support']}"
        )
    lines.append("")
    lines.append(f"accuracy,{metrics['accuracy']:.6f}")
    lines.append(f"macro_precision,{metrics['macro_precision']:.6f}")
    lines.append(f"macro_recall,{metrics['macro_recall']:.6f}")
    lines.append(f"macro_f1,{metrics['macro_f1']:.6f}")
    return "\n".join(lines)


def confusion_matrix(y_true, y_pred, class_names):
    size = len(class_names)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    return matrix


def save_confusion_matrix_csv(path, matrix, class_names):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\pred", *class_names])
        for label, row in zip(class_names, matrix):
            writer.writerow([label, *row])


def save_confusion_matrix_png(path, matrix, class_names):
    values = [value for row in matrix for value in row]
    max_value = max(values) if values else 1
    cell = 86
    left = 150
    top = 110
    width = left + cell * len(class_names) + 40
    height = top + cell * len(class_names) + 80

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        label_font = ImageFont.truetype("arial.ttf", 16)
        cell_font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        cell_font = ImageFont.load_default()

    def text_center(box, text, font, fill="black"):
        x1, y1, x2, y2 = box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text(
            (x1 + (x2 - x1 - text_width) / 2, y1 + (y2 - y1 - text_height) / 2),
            text,
            font=font,
            fill=fill,
        )

    draw.text((left, 25), "LayoutLMv3 Confusion Matrix", font=title_font, fill="black")
    draw.text(
        (left + cell * len(class_names) / 2 - 50, 70),
        "Predicted label",
        font=label_font,
        fill="black",
    )
    draw.text((20, top + cell * len(class_names) / 2 - 10), "True label", font=label_font, fill="black")

    for column, label in enumerate(class_names):
        text_center((left + column * cell, top - 38, left + (column + 1) * cell, top), label, label_font)

    for row_index, label in enumerate(class_names):
        y1 = top + row_index * cell
        y2 = y1 + cell
        text_center((0, y1, left - 8, y2), label, label_font)

        for column_index, value in enumerate(matrix[row_index]):
            x1 = left + column_index * cell
            x2 = x1 + cell
            intensity = value / max_value if max_value else 0
            shade = int(255 - 170 * intensity)
            color = (shade, shade + int(35 * (1 - intensity)), 255)
            draw.rectangle((x1, y1, x2, y2), fill=color, outline=(80, 100, 130))
            fill = "white" if intensity > 0.65 else "black"
            text_center((x1, y1, x2, y2), str(value), cell_font, fill=fill)

    image.save(path)


def save_training_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
        "validation_macro_precision",
        "validation_macro_recall",
        "validation_macro_f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_test_results(results_dir, metrics, class_names):
    results_dir.mkdir(parents=True, exist_ok=True)

    serializable_metrics = {
        key: value
        for key, value in metrics.items()
        if key not in {"y_true", "y_pred", "ids", "confidences", "image_paths", "ocr_paths"}
    }
    (results_dir / "test_metrics.json").write_text(
        json.dumps(serializable_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (results_dir / "classification_report.txt").write_text(
        classification_report_text(metrics, class_names) + "\n",
        encoding="utf-8",
    )

    matrix = confusion_matrix(metrics["y_true"], metrics["y_pred"], class_names)
    save_confusion_matrix_csv(results_dir / "confusion_matrix.csv", matrix, class_names)
    save_confusion_matrix_png(results_dir / "confusion_matrix.png", matrix, class_names)

    with (results_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "true_label", "predicted_label", "confidence", "image_path", "ocr_path"])
        for row_id, true, pred, confidence, image_path, ocr_path in zip(
            metrics["ids"],
            metrics["y_true"],
            metrics["y_pred"],
            metrics["confidences"],
            metrics["image_paths"],
            metrics["ocr_paths"],
        ):
            writer.writerow(
                [
                    row_id,
                    class_names[true],
                    class_names[pred],
                    f"{confidence:.6f}",
                    image_path,
                    ocr_path,
                ]
            )


def target_dirs(smoke_test):
    if smoke_test:
        return MODEL_DIR / "smoke_test_best_model", RESULTS_DIR / "smoke_test"
    return MODEL_DIR / "best_model", RESULTS_DIR


def print_first_batch_dimensions(loader):
    batch = next(iter(loader))
    print("First smoke-test batch dimensions:")
    for key in ("input_ids", "attention_mask", "bbox", "pixel_values", "labels"):
        value = batch.get(key)
        if torch.is_tensor(value):
            print(f"  {key}: {tuple(value.shape)}")
        else:
            print(f"  {key}: missing")


def is_cuda_oom(error):
    message = str(error).lower()
    return "cuda" in message and "out of memory" in message


def raise_cuda_oom_hint(error):
    raise RuntimeError(
        "CUDA out-of-memory while training LayoutLMv3. "
        "Ponovno pokušajte s --batch-size 1 i većim --gradient-accumulation-steps."
    ) from error


def load_pretrained_processor_and_model(class_names, label_to_index):
    id2label = {index: label for index, label in enumerate(class_names)}
    try:
        processor = AutoProcessor.from_pretrained(MODEL_NAME, apply_ocr=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(class_names),
            id2label=id2label,
            label2id=label_to_index,
        )
    except (OSError, ImportError, ValueError) as error:
        raise RuntimeError(
            f"Could not load pretrained LayoutLMv3 model/processor '{MODEL_NAME}'. "
            "Check internet access, local Hugging Face cache, and installed dependencies."
        ) from error
    return processor, model


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be at least 1")
    if args.max_length < 1:
        raise ValueError("--max-length must be at least 1")

    set_seed(RANDOM_SEED)

    if args.smoke_test:
        args.epochs = 1
        print("Smoke test enabled: using up to 5 documents per class per split and 1 epoch.")

    class_names, label_to_index = load_label_mapping()
    rows_by_split = validate_and_prepare_splits(label_to_index)
    if args.smoke_test:
        rows_by_split = limit_for_smoke_test(rows_by_split, max_per_class=5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}")
    print(f"AMP enabled: {use_amp}")

    processor, model = load_pretrained_processor_and_model(class_names, label_to_index)

    try:
        model.to(device)
    except RuntimeError as error:
        if is_cuda_oom(error):
            raise_cuda_oom_hint(error)
        raise

    loaders = make_loaders(
        rows_by_split,
        processor,
        args.batch_size,
        args.max_length,
        label_to_index,
        args.num_workers,
    )

    if args.smoke_test:
        print_first_batch_dimensions(loaders["train"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    allowed_input_keys = model_input_keys(model)

    best_model_dir, results_dir = target_dirs(args.smoke_test)
    best_val_f1 = -1.0
    epochs_without_improvement = 0
    training_history = []
    patience = 2

    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(
                model,
                loaders["train"],
                optimizer,
                scaler,
                device,
                allowed_input_keys,
                args.gradient_accumulation_steps,
                use_amp,
                class_names,
            )
            val_metrics = evaluate(
                model,
                loaders["validation"],
                device,
                allowed_input_keys,
                class_names,
                use_amp,
            )

            history_row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_loss": val_metrics["loss"],
                "validation_accuracy": val_metrics["accuracy"],
                "validation_macro_precision": val_metrics["macro_precision"],
                "validation_macro_recall": val_metrics["macro_recall"],
                "validation_macro_f1": val_metrics["macro_f1"],
            }
            training_history.append(history_row)

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train loss: {train_metrics['loss']:.4f} | "
                f"train accuracy: {train_metrics['accuracy']:.4f} | "
                f"validation loss: {val_metrics['loss']:.4f} | "
                f"validation accuracy: {val_metrics['accuracy']:.4f} | "
                f"validation macro precision: {val_metrics['macro_precision']:.4f} | "
                f"validation macro recall: {val_metrics['macro_recall']:.4f} | "
                f"validation macro F1: {val_metrics['macro_f1']:.4f}"
            )

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                epochs_without_improvement = 0
                best_model_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_model_dir)
                processor.save_pretrained(best_model_dir)
                save_label_mapping(best_model_dir / "label_mapping.json", class_names, label_to_index)
                print(f"Saved best model to: {best_model_dir}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {patience} epochs without improvement.")
                    break

    except RuntimeError as error:
        if is_cuda_oom(error):
            raise_cuda_oom_hint(error)
        raise

    results_dir.mkdir(parents=True, exist_ok=True)
    save_training_history(results_dir / "training_history.csv", training_history)

    try:
        best_processor = AutoProcessor.from_pretrained(best_model_dir, apply_ocr=False)
        best_model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
        best_model.to(device)
        best_allowed_input_keys = model_input_keys(best_model)
        test_loader = make_loaders(
            {"test": rows_by_split["test"]},
            best_processor,
            args.batch_size,
            args.max_length,
            label_to_index,
            args.num_workers,
        )["test"]
        test_metrics = evaluate(
            best_model,
            test_loader,
            device,
            best_allowed_input_keys,
            class_names,
            use_amp,
            measure_prediction_time=True,
        )
    except RuntimeError as error:
        if is_cuda_oom(error):
            raise_cuda_oom_hint(error)
        raise

    save_test_results(results_dir, test_metrics, class_names)

    print("TEST RESULTS")
    print(f"accuracy: {test_metrics['accuracy']:.4f}")
    print(f"macro precision: {test_metrics['macro_precision']:.4f}")
    print(f"macro recall: {test_metrics['macro_recall']:.4f}")
    print(f"macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"prediction time seconds: {test_metrics['prediction_time_seconds']:.4f}")
    print(f"seconds per document: {test_metrics['seconds_per_document']:.6f}")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
