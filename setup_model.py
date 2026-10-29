"""The only application script that contacts the internet. No user text is read."""
import os

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download
from engines import Qwen3VariableDetector
from model_info import MODEL_DIR, MODEL_FILES, MODEL_ID, MODEL_REVISION

QWEN_DIR = MODEL_DIR.parent / "detector-qwen3-variable"
QWEN_FILES = ("config.json", "detector-config.json", "model.safetensors",
              "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "README.md", "LICENSE")


def download(repo_id, revision, target, files, blurb):
    print(f"Downloading {repo_id} ({blurb}) for offline use.", flush=True)
    snapshot_download(repo_id=repo_id, revision=revision,
                      local_dir=str(target), allow_patterns=list(files))
    missing = [name for name in files if not (target / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete download: {', '.join(missing)}")


if __name__ == "__main__":
    qwen = Qwen3VariableDetector()
    download(MODEL_ID, MODEL_REVISION, MODEL_DIR, MODEL_FILES, "about 270 MB")
    download(qwen.engine_id, qwen.revision, QWEN_DIR, QWEN_FILES,
             "about 1.2 GB, experimental detector")
    print("Models saved locally. You can now disconnect from the internet.", flush=True)