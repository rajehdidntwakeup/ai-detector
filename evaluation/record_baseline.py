"""Record the installed baseline detector manifest without changing inference."""
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

from model_info import MAX_CHARACTERS, MAX_WORDS, MODEL_DIR, MODEL_ID, MODEL_REVISION

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    config_path = MODEL_DIR / "config.json"
    detector_config_path = MODEL_DIR / "detector-config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    detector_config = json.loads(detector_config_path.read_text(encoding="utf-8")) if detector_config_path.is_file() else {}

    model_files = {}
    for name in ("model.safetensors", "tokenizer.json", "tokenizer_config.json", "config.json", "detector-config.json"):
        path = MODEL_DIR / name
        model_files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size} if path.is_file() else None

    manifest = {
        "schemaVersion": 1,
        "kind": "baseline",
        "recordedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "detector": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "localDir": str(MODEL_DIR.relative_to(ROOT)).replace("\\", "/"),
            "baseModel": detector_config.get("base_model"),
            "trainingDataset": detector_config.get("dataset"),
            "maxLength": detector_config.get("max_length"),
            "maxPositionEmbeddings": config.get("max_position_embeddings"),
            "labelMapping": detector_config.get("label_mapping"),
            "id2label": config.get("id2label"),
            "publisherDecisionThreshold": detector_config.get("decision_threshold"),
            "calibration": detector_config.get("calibration"),
            "temperature": detector_config.get("temperature"),
            "modelFiles": model_files,
        },
        "appInference": {
            "maxWords": MAX_WORDS,
            "maxCharacters": MAX_CHARACTERS,
            "leaningThresholds": {"human": "<=0.35", "uncertain": "0.35..0.65", "ai": ">=0.65"},
            "passageSpanLimit": 80,
            "maxTokensPerWindow": min(512, int(detector_config.get("max_length", 512))),
            "specialTokensAdded": True,
            "windowOverlap": False,
            "aggregation": "token-weighted mean of document windows; passage scores computed separately",
            "inputNormalization": "strip whitespace; require >=10 characters, alphabetic character, <=maxWords and <=maxCharacters",
            "batchSize": 4,
            "torchThreads": "max(1, min(4, cpu_count // 2))",
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "transformers": importlib.metadata.version("transformers"),
            "torch": importlib.metadata.version("torch"),
            "safetensors": importlib.metadata.version("safetensors"),
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        },
        "baselineResult": "evaluation/results/baseline.json",
        "note": "Baseline manifest for rollback and paired comparisons. No thresholds or model behavior changed.",
    }

    output = Path(__file__).with_name("baseline_manifest.json")
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "detector": manifest["detector"]["id"], "revision": manifest["detector"]["revision"]}))


if __name__ == "__main__":
    main()
