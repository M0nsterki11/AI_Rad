import argparse
import json
import time
from pathlib import Path

import fitz
import torch
from PIL import Image
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "resnet50" / "best_model.pth"
LABEL_MAPPING_PATH = PROJECT_ROOT / "models" / "resnet50" / "label_mapping.json"
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}


def load_label_mapping():
    if not LABEL_MAPPING_PATH.exists():
        raise FileNotFoundError(f"Missing label mapping: {LABEL_MAPPING_PATH}")

    mapping = json.loads(LABEL_MAPPING_PATH.read_text(encoding="utf-8"))
    class_names = mapping.get("class_names")
    label_to_index = mapping.get("label_to_index")

    if not isinstance(class_names, list) or not class_names:
        raise ValueError("label_mapping.json must contain a non-empty class_names list.")

    if not isinstance(label_to_index, dict):
        raise ValueError("label_mapping.json must contain label_to_index.")

    index_to_label = {int(index): label for label, index in label_to_index.items()}
    ordered_labels = [index_to_label[index] for index in sorted(index_to_label)]

    if ordered_labels != class_names:
        raise ValueError(
            "label_mapping.json class_names order does not match label_to_index order."
        )

    return class_names, label_to_index


def inspect_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        checkpoint_class_names = checkpoint.get("class_names")
        checkpoint_label_to_index = checkpoint.get("label_to_index")
        structure = {
            "type": "training_checkpoint",
            "keys": sorted(checkpoint.keys()),
            "epoch": checkpoint.get("epoch"),
            "validation_macro_f1": checkpoint.get("validation_macro_f1"),
        }
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        checkpoint_class_names = None
        checkpoint_label_to_index = None
        structure = {
            "type": "raw_state_dict",
            "keys_sample": list(checkpoint.keys())[:10],
        }
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint model_state_dict is not a dict.")

    if "fc.weight" not in state_dict or "fc.bias" not in state_dict:
        raise ValueError("Checkpoint is missing fc.weight or fc.bias.")

    output_count = int(state_dict["fc.bias"].shape[0])
    input_count = int(state_dict["fc.weight"].shape[1])

    structure["fc_weight_shape"] = list(state_dict["fc.weight"].shape)
    structure["fc_bias_shape"] = list(state_dict["fc.bias"].shape)
    structure["checkpoint_class_names"] = checkpoint_class_names
    structure["checkpoint_label_to_index"] = checkpoint_label_to_index

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


def render_pdf_first_page(path, zoom=2.0):
    document = fitz.open(str(path))
    try:
        if document.page_count < 1:
            raise ValueError(f"PDF has no pages: {path}")

        page = document.load_page(0)
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        return image
    finally:
        document.close()


def load_document_image(path):
    path = Path(path)
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file extension '{extension}'. Supported: {supported}")

    if extension == ".pdf":
        return render_pdf_first_page(path).convert("RGB")

    with Image.open(path) as image:
        return image.convert("RGB")


def load_model(device):
    class_names, label_to_index = load_label_mapping()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict, output_count, input_count, structure = inspect_checkpoint(checkpoint)

    if output_count != len(class_names):
        raise ValueError(
            f"Checkpoint output count is {output_count}, but label mapping has {len(class_names)} classes."
        )

    checkpoint_label_to_index = structure.get("checkpoint_label_to_index")
    if checkpoint_label_to_index is not None and checkpoint_label_to_index != label_to_index:
        raise ValueError(
            "Checkpoint label_to_index does not match models/resnet50/label_mapping.json."
        )

    model = models.resnet50(weights=None)
    if input_count != model.fc.in_features:
        raise ValueError(
            f"Checkpoint fc input size is {input_count}, expected {model.fc.in_features}."
        )

    model.fc = torch.nn.Linear(model.fc.in_features, output_count)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, class_names, structure


@torch.no_grad()
def predict_file(path, model=None, class_names=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_structure = None
    if model is None or class_names is None:
        model, class_names, checkpoint_structure = load_model(device)

    preprocess = make_preprocess()
    image = load_document_image(path)
    tensor = preprocess(image).unsqueeze(0).to(device)

    start = time.perf_counter()
    logits = model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    prediction_time = time.perf_counter() - start

    probabilities_tensor = torch.softmax(logits, dim=1).squeeze(0).cpu()
    probabilities = {
        label: float(probabilities_tensor[index].item())
        for index, label in enumerate(class_names)
    }
    predicted_index = int(probabilities_tensor.argmax().item())
    predicted_class = class_names[predicted_index]
    confidence = probabilities[predicted_class]

    return {
        "file": str(Path(path)),
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "prediction_time_seconds": prediction_time,
        "device": str(device),
        "checkpoint_structure": checkpoint_structure,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a document class with trained ResNet50.")
    parser.add_argument("--file", required=True, help="Path to a PDF, PNG, JPG, or JPEG document.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, checkpoint_structure = load_model(device)
    result = predict_file(args.file, model=model, class_names=class_names, device=device)
    result["checkpoint_structure"] = checkpoint_structure
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
