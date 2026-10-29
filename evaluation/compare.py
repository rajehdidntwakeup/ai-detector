"""Side-by-side detector comparison on the manifest. Run offline."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import socket

from engines import ENGINES
from evaluation.run import load_manifest
from evaluation.memory import peak_rss_bytes
from engines.inputs import ContextLimitError

ROOT = Path(__file__).resolve().parents[1]


def implementation_hash():
    digest = hashlib.sha256()
    for name in ("detector.py", "engines/__init__.py", "engines/inputs.py", "evaluation/compare.py"):
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def leaning(score):
    if score >= 0.65:
        return "ai"
    if score <= 0.35:
        return "human"
    return "uncertain"


def deny_network(*args, **kwargs):
    raise RuntimeError("Evaluation attempted a network connection")


def run_engine(name, records):
    cls = ENGINES[name]
    adapter = cls()
    started = time.perf_counter()
    adapter.load()
    load_ms = round((time.perf_counter() - started) * 1000)
    peak_load = peak_rss_bytes()
    rows = []
    for record in records:
        t0 = time.perf_counter()
        try:
            score = adapter.score_text(record["text"])
            outcome = {"status": "ok", "score": score, "verdict": leaning(score), "truncated": False}
        except ContextLimitError as exc:
            outcome = {"status": "unsupported_length", "score": None, "verdict": None,
                       "inputTokens": exc.tokens, "tokenLimit": exc.limit, "analyzedTokens": 0,
                       "truncated": False, "message": str(exc)}
        duration_ms = round((time.perf_counter() - t0) * 1000)
        rows.append({
            "id": record["id"], "label": record["label"], "split": record["split"], "group": record["group"],
            "sha256": hashlib.sha256(record["text"].encode()).hexdigest(),
            "wordCount": len(record["text"].split()), "durationMs": duration_ms, **outcome,
        })
    return {
        "engine": adapter.to_engine_record(),
        "loadMs": load_ms, "afterLoadPeakRssBytes": peak_load, "peakProcessRssBytes": peak_rss_bytes(),
        "memoryMethod": "OS process lifetime high-water RSS; one fresh worker per engine",
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engines", nargs="+", choices=sorted(ENGINES), default=["baseline", "modernbert"])
    parser.add_argument("--worker", choices=sorted(ENGINES), help=argparse.SUPPRESS)
    args = parser.parse_args()
    records = load_manifest(args.manifest)
    if args.worker:
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
        with patch.object(socket.socket, "connect", deny_network), patch.object(socket.socket, "connect_ex", deny_network), patch("socket.create_connection", deny_network):
            print(json.dumps(run_engine(args.worker, records)))
        return
    if len(set(args.engines)) != len(args.engines):
        parser.error("Specify each engine once")
    engines = []
    for name in args.engines:
        worker = subprocess.run([sys.executable, "-m", "evaluation.compare", "--worker", name,
                                 "--manifest", str(args.manifest.resolve()), "--output", str(args.output.resolve())],
                                cwd=ROOT, capture_output=True, text=True, timeout=1800)
        if worker.returncode:
            raise RuntimeError(f"{name} evaluation failed: {worker.stderr[-3000:]}")
        engines.append(json.loads(worker.stdout))
        print(f"Completed {name}", flush=True)
    report = {
        "schemaVersion": 2,
        "implementationSha256": implementation_hash(),
        "python": platform.python_version(),
        "transformers": importlib.metadata.version("transformers"),
        "torch": importlib.metadata.version("torch"),
        "manifestSha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "offlineVerified": True,
        "engines": engines,
        "notice": "Comparison on limited screening data; no thresholds tuned. Passage-level scores not reported. Mixed/edited samples excluded from binary metrics.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "engines": [e["engine"]["id"] for e in engines]}))


if __name__ == "__main__":
    main()
