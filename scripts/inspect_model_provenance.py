"""Safe file/tensor comparisons and unauthenticated public HF metadata only."""
from __future__ import annotations
import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_models import sha, dump, rows
from scripts.upload_models_to_hf import DEFAULT_REPOS


def main():
    import torch
    from safetensors import safe_open
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((ROOT / "reports").resolve()):
        parser.error("Output must be in reports")
    result = {"utc": datetime.now(timezone.utc).isoformat(), "remote_is_not_cloud_runtime_evidence": True, "models": {}}
    for key in DEFAULT_REPOS:
        record = {}
        relative = "best_model.pth" if key == "resnet50" else "best_model/model.safetensors"
        current = ROOT / "models" / f"{key}_multipage" / relative
        legacy = ROOT / "models" / key / relative
        record["current"] = {"path": str(current), "sha256": sha(current), "mtime_utc": datetime.fromtimestamp(current.stat().st_mtime, timezone.utc).isoformat()}
        record["legacy"] = {"path": str(legacy), "sha256": sha(legacy)} if legacy.is_file() else None
        record["history"] = rows(ROOT / "results" / f"{key}_multipage/training_history.csv")
        record["test_metrics"] = json.loads((ROOT / "results" / f"{key}_multipage/test_metrics.json").read_text())
        # These are upload-script defaults, NOT a claim about runtime secrets.
        repo = DEFAULT_REPOS[key]
        remote_path = "model.safetensors" if key == "layoutlmv3" else relative
        try:
            url = f"https://huggingface.co/api/models/{repo}?blobs=true"
            with urllib.request.urlopen(url, timeout=20) as response:
                info = json.load(response)
            remote_file = next((f for f in info.get("siblings", []) if f["rfilename"] == remote_path), {})
            remote_sha = remote_file.get("lfs", {}).get("sha256")
            record["public_upload_repo"] = {"repo_id": repo, "revision": info.get("sha"), "last_modified": info.get("lastModified"),
                "weight_path": remote_path, "weight_sha256": remote_sha, "matches_local_bytes": remote_sha == record["current"]["sha256"] if remote_sha else None}
        except Exception as exc:
            record["public_upload_repo"] = {"repo_id": repo, "status": "unavailable", "error_type": type(exc).__name__}
        if key == "resnet50":
            base = Path.home() / ".cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth"
            current_state = torch.load(current, map_location="cpu", weights_only=True)["model_state_dict"]
            original = torch.load(base, map_location="cpu", weights_only=True)
            common = set(current_state) & set(original)
            different_shape = [k for k in common if current_state[k].shape != original[k].shape]
            equal = [k for k in common - set(different_shape) if torch.equal(current_state[k], original[k])]
            unequal = sorted(common - set(different_shape) - set(equal))
            record["base_comparison"] = {"cached_base_path": str(base), "base_sha256": sha(base), "equal_keys": sorted(equal),
                "changed_keys": unequal, "different_shapes": different_shape}
            del current_state, original
        else:
            name = "models--xlm-roberta-base" if key == "xlm_roberta" else "models--microsoft--layoutlmv3-base"
            candidates = sorted((Path.home() / ".cache/huggingface/hub" / name / "snapshots").glob("*/model.safetensors"))
            if candidates:
                base = candidates[-1]
                with safe_open(current, framework="pt") as current_state, safe_open(base, framework="pt") as original:
                    current_keys, base_keys = set(current_state.keys()), set(original.keys())
                    common = current_keys & base_keys
                    equal = [k for k in common if torch.equal(current_state.get_tensor(k), original.get_tensor(k))]
                    record["base_comparison"] = {"cached_base_path": str(base), "cached_revision_not_historical_proof": base.parent.name,
                        "base_sha256": sha(base), "common_keys": len(common), "equal_keys": sorted(equal), "changed_count": len(common) - len(equal),
                        "only_saved_keys": sorted(current_keys-base_keys), "only_base_keys": sorted(base_keys-current_keys)}
        result["models"][key] = record
        print(key, record["current"]["sha256"], record["public_upload_repo"], flush=True)
    dump(args.output, result)


if __name__ == "__main__":
    main()
