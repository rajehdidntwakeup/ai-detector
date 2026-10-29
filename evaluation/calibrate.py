"""Calibrate thresholds on the calibration split and evaluate on held-out data."""
import argparse
import json
import math
import hashlib
from pathlib import Path

from evaluation.run import load_manifest, rate
from evaluation.compare import implementation_hash


def leaning(score, thresholds):
    if score >= thresholds["ai"]:
        return "ai"
    if thresholds["human"] is not None and score <= thresholds["human"]:
        return "human"
    return "uncertain"


def binary_metrics(rows, thresholds, human_label="human", ai_label="ai"):
    human = [r for r in rows if r["label"] == human_label]
    ai = [r for r in rows if r["label"] == ai_label]
    scored = [r for r in rows if r["label"] in (human_label, ai_label)]
    return {
        "n": len(scored),
        "falsePositiveRate": rate(sum(leaning(r["score"], thresholds) == "ai" for r in human), len(human)),
        "aiRecall": rate(sum(leaning(r["score"], thresholds) == "ai" for r in ai), len(ai)),
        "uncertainRate": rate(sum(leaning(r["score"], thresholds) == "uncertain" for r in scored), len(scored)),
    }


def brier(rows, human_label="human", ai_label="ai"):
    items = [r for r in rows if r["label"] in (human_label, ai_label)]
    if not items:
        return None
    total = 0.0
    for r in items:
        y = 1 if r["label"] == ai_label else 0
        total += (r["score"] - y) ** 2
    return total / len(items)


def ece(rows, bins=10, human_label="human", ai_label="ai"):
    items = [r for r in rows if r["label"] in (human_label, ai_label)]
    if not items:
        return None
    boundaries = [i / bins for i in range(bins + 1)]
    total = 0.0
    count = 0
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        bucket = [r for r in items if lo <= r["score"] < hi or (hi == 1 and r["score"] == 1)]
        if not bucket:
            continue
        accuracy = sum(r["label"] == ai_label for r in bucket) / len(bucket)
        confidence = sum(r["score"] for r in bucket) / len(bucket)
        total += len(bucket) * abs(accuracy - confidence)
        count += len(bucket)
    return total / count if count else None


def find_thresholds(rows, fp_target=0.05):
    """Fit both tails with bounded empirical error rates; no silent fallback."""
    if not math.isfinite(fp_target) or not 0 <= fp_target <= 1:
        raise ValueError("False-positive target must be between 0 and 1")
    if any(not isinstance(r["score"], (int, float)) or not math.isfinite(r["score"]) or not 0 <= r["score"] <= 1 for r in rows):
        raise ValueError("Scores must be finite numbers between 0 and 1")
    human = [r for r in rows if r["label"] == "human"]
    ai = [r for r in rows if r["label"] == "ai"]
    if not human or not ai:
        return {"status": "insufficient_classes", "ai": None, "human": None}
    observed = {r["score"] for r in human + ai}
    scores = sorted({0.0, 1.0, *observed, *(math.nextafter(s, 1.0) for s in observed)})
    best = None
    for ai_thresh in scores:
        fp = sum(r["score"] >= ai_thresh for r in human) / len(human)
        if fp > fp_target:
            continue
        rec = sum(r["score"] >= ai_thresh for r in ai) / len(ai)
        if best is None or (rec, -fp, ai_thresh) > (best["aiRecall"], -best["falsePositiveRate"], best["ai"]):
            best = {"status": "fitted", "ai": float(ai_thresh), "human": None,
                    "aiRecall": float(rec), "falsePositiveRate": float(fp), "falseHumanRate": None}
    if best is None:
        return {"status": "no_feasible_operating_point", "ai": None, "human": None}
    lower_candidates = {0.0, *observed, *(math.nextafter(s, 0.0) for s in observed)}
    best_human_recall = -1
    for threshold in sorted(t for t in lower_candidates if 0 <= t < best["ai"]):
        false_human = sum(r["score"] <= threshold for r in ai) / len(ai)
        human_recall = sum(r["score"] <= threshold for r in human) / len(human)
        if false_human <= fp_target and human_recall > best_human_recall:
            best.update(human=threshold, falseHumanRate=false_human)
            best_human_recall = human_recall
    return best


def validate_comparison(compare, manifest, manifest_hash):
    if compare.get("schemaVersion") != 2 or compare.get("implementationSha256") != implementation_hash():
        raise ValueError("Comparison predates the corrected inference implementation; rerun it")
    if compare.get("manifestSha256") != manifest_hash:
        raise ValueError("Comparison manifest hash does not match")
    expected = {r["id"]: r for r in manifest}
    seen_engines = set()
    if not compare.get("engines"):
        raise ValueError("Comparison contains no engines")
    for engine in compare["engines"]:
        identity = engine["engine"]["id"]
        if identity in seen_engines:
            raise ValueError("Duplicate engine")
        seen_engines.add(identity)
        rows = engine["rows"]
        if len(rows) != len(expected) or {r["id"] for r in rows} != set(expected):
            raise ValueError("Comparison must cover each manifest ID exactly once")
        for row in rows:
            original = expected[row["id"]]
            if row["sha256"] != hashlib.sha256(original["text"].encode()).hexdigest() or any(row[k] != original[k] for k in ("label", "split", "group")):
                raise ValueError("Comparison row identity mismatch")
            score = row.get("score")
            if row.get("status") != "ok" or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Every comparison case must have a valid score; resolve unsupported inputs before calibration")


def subgroup_report(rows, thresholds):
    groups = {}
    for r in rows:
        for dim in ("domain", "citationDensity", "generatorFamily"):
            key = r.get(dim, "unknown")
            groups.setdefault(dim, {}).setdefault(key, []).append(r)
    report = {}
    for dim, buckets in groups.items():
        report[dim] = {key: binary_metrics(val, thresholds) for key, val in buckets.items()}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.jsonl"))
    parser.add_argument("--compare", type=Path, required=True, help="evaluation/results/compare-all-four.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fp-target", type=float, default=0.05)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    compare = json.loads(args.compare.read_text(encoding="utf-8"))
    validate_comparison(compare, manifest, hashlib.sha256(args.manifest.read_bytes()).hexdigest())

    # Map scores by id for each engine.
    by_engine = {}
    for engine in compare["engines"]:
        by_engine[engine["engine"]["id"]] = {r["id"]: r["score"] for r in engine["rows"]}

    records_by_split = {split: [r for r in manifest if r["split"] == split] for split in {"development", "calibration", "heldout"}}

    calibrated = []
    for engine in compare["engines"]:
        engine_id = engine["engine"]["id"]
        scores = by_engine[engine_id]

        def attach(split):
            out = []
            for r in records_by_split[split]:
                if r["id"] not in scores:
                    continue
                out.append(dict(r, score=scores[r["id"]]))
            return out

        cal_rows = attach("calibration")
        thresholds = find_thresholds(cal_rows, args.fp_target)
        if thresholds["status"] != "fitted":
            raise ValueError(f"{engine_id}: {thresholds['status']}; no thresholds published")

        dev_rows = attach("development")
        held_rows = attach("heldout")
        calibrated.append({
            "engine": engine["engine"],
            "loadMs": engine["loadMs"],
            "afterLoadPeakRssBytes": engine["afterLoadPeakRssBytes"],
            "peakProcessRssBytes": engine["peakProcessRssBytes"],
            "thresholds": thresholds,
            "calibration": binary_metrics(cal_rows, thresholds) | {"brier": brier(cal_rows), "ece": ece(cal_rows)},
            "development": binary_metrics(dev_rows, thresholds) | {"brier": brier(dev_rows), "ece": ece(dev_rows)},
            "heldout": binary_metrics(held_rows, thresholds) | {"brier": brier(held_rows), "ece": ece(held_rows)},
            "subgroups": {
                "development": subgroup_report(dev_rows, thresholds),
                "calibration": subgroup_report(cal_rows, thresholds),
                "heldout": subgroup_report(held_rows, thresholds),
            },
            "perCaseHeldout": [
                {"id": r["id"], "label": r["label"], "score": r["score"], "verdict": leaning(r["score"], thresholds),
                 "domain": r.get("domain"), "generatorFamily": r.get("generatorFamily")}
                for r in held_rows
            ],
        })

    report = {
        "schemaVersion": 2,
        "comparisonSha256": hashlib.sha256(args.compare.read_bytes()).hexdigest(),
        "manifestSha256": compare["manifestSha256"],
        "implementationSha256": compare["implementationSha256"],
        "fpTarget": args.fp_target,
        "engines": calibrated,
        "promotion": {
            "baselineAIRecallOnHeldout": next((e["heldout"]["aiRecall"]["rate"] for e in calibrated if e["engine"]["id"] == "rasbt/ai-text-detector-distilbert"), None),
        },
        "notice": "Calibration thresholds fitted on calibration split; all metrics on held-out split are primary. Small corpus; intervals are wide.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output)}))


if __name__ == "__main__":
    main()
