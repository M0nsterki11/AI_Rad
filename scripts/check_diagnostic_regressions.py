"""Small synthetic reproductions: no real model loads or document uploads."""
from __future__ import annotations
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def checks():
    import fitz
    import app
    from src import model_downloader as downloader, predict_resnet, predict_text_model, predict_layoutlm
    from src.multipage_preprocess import _clean_pdf_words
    from src.preprocess import render_pdf_page
    result = {}
    rotation = []
    with fitz.open() as doc:
        page = doc.new_page(width=400, height=600)
        page.insert_text((45, 80), "DIAGNOSTIC", fontsize=20)
        for angle in [0, 90, 180, 270]:
            page.set_rotation(angle)
            image = render_pdf_page(page)
            words, boxes = _clean_pdf_words(page, image)
            rectangle = fitz.Rect(page.get_text("words")[0][:4]) * page.rotation_matrix
            expected = [round(rectangle.x0*2), round(rectangle.y0*2), round(rectangle.x1*2), round(rectangle.y1*2)]
            rotation.append(dict(rotation=angle, actual=boxes[0], expected=expected, correct=boxes[0] == expected))
    result["rotated_pdf_boxes"] = rotation
    with tempfile.TemporaryDirectory(prefix="document_ai_regression_") as temp:
        root = Path(temp)
        multi = root / "models/resnet50_multipage"
        multi.mkdir(parents=True)
        weight = multi / "best_model.pth"
        weight.write_bytes(b"synthetic_checkpoint_1")
        (multi / "label_mapping.json").write_text("{}")
        loads = []
        def fake_load(device):
            loads.append(weight.read_bytes().decode())
            return object(), ["invoice"], {}
        app.get_resnet_model.clear()
        with patch.object(app, "load_resnet_model", side_effect=fake_load), patch.object(predict_resnet, "MULTIPAGE_MODEL_PATH", weight):
            first = app.get_resnet_model()[0]
            same = app.get_resnet_model()[0]
            weight.write_bytes(b"synthetic_checkpoint_2_updated")
            changed = app.get_resnet_model()[0]
        app.get_resnet_model.clear()
        result["cache"] = dict(unchanged_reuses_object=first is same, updated_reloads_object=first is not changed, loader_calls=len(loads))
        for name, module in [("xlm_roberta", predict_text_model), ("layoutlmv3", predict_layoutlm)]:
            incomplete = root / f"models/{name}_multipage/best_model"
            legacy = root / f"models/{name}/best_model"
            incomplete.mkdir(parents=True)
            legacy.mkdir(parents=True)
            (incomplete / "config.json").write_text("{}")
            for filename in ["config.json", "model.safetensors", "tokenizer.json", "label_mapping.json"]:
                (legacy / filename).write_text("{}")
            with patch.object(module, "MULTIPAGE_MODEL_DIR", incomplete), patch.object(module, "LEGACY_MODEL_DIR", legacy), patch.object(downloader, "get_project_root", return_value=root):
                result[name + "_incomplete_fallback"] = dict(availability=downloader.is_model_available(name),
                    selected_complete_legacy=module.active_model_dir() == legacy)
        snapshot = root / "snapshot"
        snapshot.mkdir()
        (snapshot / "best_model.pth").write_bytes(b"synthetic_raw_state_dict")
        (snapshot / "label_mapping.json").write_text("{}")
        aggregation = {"method": "mean", "top_k": 2}
        (snapshot / "aggregation_config.json").write_text(json.dumps(aggregation))
        with patch.object(downloader, "_snapshot_download", return_value=snapshot), patch.object(downloader, "get_project_root", return_value=root):
            target = downloader.download_resnet50()
            config = target / "aggregation_config.json"
            result["resnet_download_aggregation_preserved"] = config.is_file() and json.loads(config.read_text()) == aggregation
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to((ROOT / "reports").resolve()):
        parser.error("Output must be in reports")
    result = checks()
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
