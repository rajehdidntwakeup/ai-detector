"""Explicit network setup for detector candidates. Never imports user documents."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", choices=("modernbert", "desklib", "qwen3-variable"))
    args = parser.parse_args()
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import snapshot_download
    candidate = json.loads(Path(__file__).with_name("detector_candidates.json").read_text(encoding="utf-8"))[args.candidate]
    local_dir = ROOT / "models" / ("detector-" + args.candidate)
    print(f"Downloading {candidate['id']}@{candidate['revision']} for offline use.", flush=True)
    snapshot_download(
        candidate["id"],
        revision=candidate["revision"],
        local_dir=str(local_dir),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "README.md", "LICENSE*", "merges.txt"],
    )
    print("Downloaded pinned detector candidate", args.candidate)


if __name__ == "__main__":
    main()
