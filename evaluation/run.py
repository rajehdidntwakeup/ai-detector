"""Run: python -m evaluation.run --output evaluation/results/baseline.json"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import socket


def load_manifest(path):
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("Manifest is empty")
    ids, hashes, groups = set(), set(), {}
    for record in records:
        for key in ("id", "text", "label", "source", "permission", "group", "split"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f"Missing or invalid {key}")
        if record["label"] not in {"human", "ai", "mixed", "edited"}:
            raise ValueError("Unknown provenance label")
        if record["split"] not in {"regression", "development", "calibration", "heldout"}:
            raise ValueError("Unknown split")
        digest = hashlib.sha256(record["text"].strip().encode()).hexdigest()
        if record["id"] in ids or digest in hashes:
            raise ValueError("Duplicate document or ID")
        if groups.setdefault(record["group"], record["split"]) != record["split"]:
            raise ValueError("Source group leaks across splits")
        ids.add(record["id"])
        hashes.add(digest)
    return records


def rate(hits, total):
    if not total:
        return {"count": hits, "n": 0, "rate": None, "wilson95": None}
    p, z = hits / total, 1.96
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return {"count": hits, "n": total, "rate": p, "wilson95": [max(0, center-radius), min(1, center+radius)]}


def summarize(rows):
    human = [r for r in rows if r["label"] == "human"]
    ai = [r for r in rows if r["label"] == "ai"]
    return {"n": len(rows), "labels": dict(Counter(r["label"] for r in rows)),
            "falsePositiveRate": rate(sum(r["verdict"] == "ai" for r in human), len(human)),
            "aiRecall": rate(sum(r["verdict"] == "ai" for r in ai), len(ai)),
            "uncertainRate": rate(sum(r["verdict"] == "uncertain" for r in rows), len(rows))}


def deny_network(*args, **kwargs):
    raise RuntimeError("Evaluation attempted a network connection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = load_manifest(args.manifest)
    with patch.object(socket.socket, "connect", deny_network), patch.object(socket.socket, "connect_ex", deny_network), patch("socket.create_connection", deny_network):
        from detector import Detector
        started = time.perf_counter()
        engine = Detector()
        load_ms = round((time.perf_counter()-started)*1000)
        rows = []
        for record in records:
            result = engine.analyze(record["text"])
            rows.append({k: record[k] for k in ("id", "label", "split", "group") } | {
                "sha256": hashlib.sha256(record["text"].encode()).hexdigest(),
                "score": result["aiConfidence"], "verdict": result["verdict"],
                "words": result["wordCount"], "durationMs": result["durationMs"]})
        report = {"schemaVersion": 1, "engine": result["engine"], "python": platform.python_version(),
                  "transformers": importlib.metadata.version("transformers"), "torch": importlib.metadata.version("torch"),
                  "manifestSha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  "offlineVerified": True, "loadMs": load_ms, "rows": rows,
                  "bySplit": {split: summarize([r for r in rows if r["split"] == split]) for split in sorted({r["split"] for r in rows})},
                  "notice": "Regression data is not a held-out accuracy benchmark. Missing classes have null metrics. Mixed/edited samples are excluded from binary recall and false positives."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "bySplit": report["bySplit"]}))


if __name__ == "__main__":
    main()
