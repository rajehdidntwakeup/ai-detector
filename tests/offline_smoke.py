"""Real inference with all outbound socket connections denied. Run separately."""
import json
import math
from pathlib import Path
import socket
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def blocked(*args, **kwargs):
    raise AssertionError("Offline test attempted an outbound network connection.")


if __name__ == "__main__":
    with patch.object(socket.socket, "connect", blocked), patch.object(socket.socket, "connect_ex", blocked), patch("socket.create_connection", blocked):
        from detector import Detector
        detector = Detector()
        text = "Dr. Smith walked to the station. 😀 The train was late.\n\nThe train was late. " + (
            "I remembered the umbrella this time, but left my lunch on the kitchen counter. " * 45
        )
        result = detector.analyze(text)
        assert 0 <= result["aiConfidence"] <= 1
        assert math.isclose(result["aiConfidence"] + result["humanRate"], 1)
        assert "".join(part["text"] for part in result["segments"]) == text.strip()
        assert all(0 <= passage["aiScore"] <= 1 for passage in result["passages"])
        assert result["wordCount"] == len(text.split())
        assert result["engine"]["offline"] is True
        token_count = len(detector.tokenizer.encode(text.strip(), add_special_tokens=False, verbose=False))
        assert result["coverage"]["textTokensTotal"] == token_count
        assert result["coverage"]["textTokensAnalyzed"] == token_count
        assert result["coverage"]["documentWindows"] > 1
        assert result["coverage"]["truncated"] is False
        assert len(detector.tokenizer.encode(text, verbose=False)) > 512  # Exercise multiple windows.
        print(json.dumps({"offlineInference": "passed", "wordCount": result["wordCount"],
                          "passages": len(result["passages"]), "durationMs": result["durationMs"],
                          "aiScore": result["aiConfidence"]}))
