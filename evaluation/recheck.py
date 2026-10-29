"""Recheck completed trial candidates locally, without treating score changes as quality."""
import argparse
import json
from pathlib import Path
import socket
from unittest.mock import patch
from evaluation.run import deny_network


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trial = json.loads(args.trial.read_text(encoding="utf-8"))
    if args.output.resolve() == args.trial.resolve():
        parser.error("Comparison must not overwrite the trial report")
    rows = []
    with patch.object(socket.socket, "connect", deny_network), patch.object(socket.socket, "connect_ex", deny_network), patch("socket.create_connection", deny_network):
        from detector import Detector
        engine = Detector()
        for row in trial["rows"]:
            if not row["complete"]:
                rows.append({"id": row["id"], "status": "skipped_incomplete"})
                continue
            try:
                original = engine.analyze(row["input"])
                revised = engine.analyze(row["output"])
            except ValueError as exc:
                rows.append({"id": row["id"], "status": "invalid_text", "message": str(exc)})
                continue
            rows.append({"id": row["id"], "status": "compared", "engine": original["engine"],
                         "originalScore": original["aiConfidence"], "revisedScore": revised["aiConfidence"],
                         "delta": revised["aiConfidence"] - original["aiConfidence"],
                         "originalWords": original["wordCount"], "revisedWords": revised["wordCount"]})
    result = {"rewriter": trial["engine"], "offlineVerified": True, "rows": rows,
              "notice": "A lower detector score does not establish better writing, preserved meaning, human authorship, or agreement with other detectors."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
