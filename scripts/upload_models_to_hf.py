from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOS = {
    "resnet50": "M0nsterki11/document-ai-resnet50",
    "xlm_roberta": "M0nsterki11/document-ai-xlm-roberta",
    "layoutlmv3": "M0nsterki11/document-ai-layoutlmv3",
}
UPLOAD_TARGETS = {
    "resnet50": PROJECT_ROOT / "models" / "resnet50",
    "xlm_roberta": PROJECT_ROOT / "models" / "xlm_roberta",
    "layoutlmv3": PROJECT_ROOT / "models" / "layoutlmv3" / "best_model",
}
IGNORE_PATTERNS = [
    "data/**",
    "results/**",
    "venv/**",
    ".venv/**",
    "__pycache__/**",
    "*.pyc",
    "runs/**",
    "logs/**",
    ".cache/**",
    ".hf_cache/**",
    "smoke_test_best_model/**",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload trained Document AI Classifier models to Hugging Face Hub."
    )
    parser.add_argument(
        "--resnet50-repo-id",
        default=os.environ.get("HF_RESNET50_REPO_ID", DEFAULT_REPOS["resnet50"]),
        help="Hugging Face model repo for models/resnet50/.",
    )
    parser.add_argument(
        "--xlm-roberta-repo-id",
        default=os.environ.get("HF_XLM_ROBERTA_REPO_ID", DEFAULT_REPOS["xlm_roberta"]),
        help="Hugging Face model repo for models/xlm_roberta/.",
    )
    parser.add_argument(
        "--layoutlmv3-repo-id",
        default=os.environ.get("HF_LAYOUTLMV3_REPO_ID", DEFAULT_REPOS["layoutlmv3"]),
        help="Hugging Face model repo for models/layoutlmv3/best_model/.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create Hugging Face repos as private if they do not exist.",
    )
    return parser.parse_args()


def require_token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required for uploading. Set it in your terminal environment first."
        )
    return token


def validate_folder(model_key, folder):
    if not folder.exists():
        raise FileNotFoundError(f"Missing local model folder for {model_key}: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Expected a folder for {model_key}: {folder}")


def upload_model(api, model_key, repo_id, folder, token, private):
    validate_folder(model_key, folder)
    print(f"Creating/checking repo: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
        token=token,
    )

    print(f"Uploading {model_key}: {folder}")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder),
        path_in_repo=".",
        token=token,
        ignore_patterns=IGNORE_PATTERNS,
    )
    print(f"Done: {repo_id}")


def main():
    args = parse_args()
    token = require_token()
    api = HfApi(token=token)

    repo_ids = {
        "resnet50": args.resnet50_repo_id,
        "xlm_roberta": args.xlm_roberta_repo_id,
        "layoutlmv3": args.layoutlmv3_repo_id,
    }

    for model_key, folder in UPLOAD_TARGETS.items():
        upload_model(
            api=api,
            model_key=model_key,
            repo_id=repo_ids[model_key],
            folder=folder,
            token=token,
            private=args.private,
        )


if __name__ == "__main__":
    main()
