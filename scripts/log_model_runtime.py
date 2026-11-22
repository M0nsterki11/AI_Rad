"""Print only code/model provenance. No document needed; no downloads or training.

--load uses app resource getters and verifies model.eval in the returned object.
Without --load, only selected files are fingerprinted (not a runtime proof).
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from src.model_runtime import runtime_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--model", choices=["resnet50", "xlm_roberta", "layoutlmv3"], required=True)
    args = parser.parse_args()
    model = None
    if args.load:
        import app
        getter = {"resnet50": app.get_resnet_model, "xlm_roberta": app.get_text_model, "layoutlmv3": app.get_layoutlm_model}[args.model]
        model = getter()[0]
    print(json.dumps(runtime_record(args.model, model), ensure_ascii=True, indent=2))
