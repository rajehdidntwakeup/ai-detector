"""Real model preprocessing parity and context guards, with outbound networking blocked."""
import gc
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.run import deny_network, load_manifest
from engines.inputs import bert_window, bounded_inputs, qwen_readout_inputs, ContextLimitError


def main():
    records = [r for r in load_manifest(ROOT / "evaluation/manifest.jsonl") if r["split"] == "regression"]
    with patch.object(socket.socket, "connect", deny_network), patch.object(socket.socket, "connect_ex", deny_network), patch("socket.create_connection", deny_network):
        from detector import Detector
        from engines import DistilbertDetector, Qwen3VariableDetector
        d = Detector()
        adapter = DistilbertDetector()
        for field in ("torch", "tokenizer", "model", "temperature", "ai_index", "max_tokens"):
            setattr(adapter, field, getattr(d, field))
        cases = []
        for r in records:
            inputs = d.tokenizer(r["text"], return_tensors="pt", truncation=False)
            assert inputs["input_ids"].shape[-1] <= 512
            with d.torch.inference_mode():
                expected = d.torch.softmax(d.model(**inputs).logits.float() / d.temperature, dim=-1)[0, d.ai_index].item()
            actual = d.score_texts([r["text"]])[0]
            assert abs(expected - actual) < 1e-6, (r["id"], expected, actual)
            assert abs(expected - adapter.score_text(r["text"])) < 1e-6
            cases.append({"id": r["id"], "aiScore": actual})
        capacity = 510
        ids = d.tokenizer.encode("word " * 1100, add_special_tokens=False)
        recovered = []
        weighted = 0
        for offset in range(0, len(ids), capacity):
            chunk = ids[offset:offset+capacity]
            prepared = bert_window(d.tokenizer, chunk)
            assert len(prepared["input_ids"]) <= 512
            assert prepared["input_ids"][0] == d.tokenizer.cls_token_id
            assert prepared["input_ids"][-1] == d.tokenizer.sep_token_id
            recovered += prepared["input_ids"][1:-1]
            tensors = {k: d.torch.tensor([v]) for k, v in prepared.items()}
            with d.torch.inference_mode():
                score = d.torch.softmax(d.model(**tensors).logits.float()/d.temperature, dim=-1)[0, d.ai_index].item()
            weighted += len(chunk) * score
        assert recovered == ids
        actual = d.score_texts(["word " * 1100])[0]
        assert abs(actual - weighted/len(ids)) < 1e-6
        try:
            bounded_inputs(d.tokenizer, "word " * 1100, 512)
        except ContextLimitError as exc:
            assert exc.tokens > exc.limit
        else:
            raise AssertionError("Expected explicit context rejection")
        del adapter, d
        gc.collect()
        q = Qwen3VariableDetector()
        q.load()
        text = records[-1]["text"]
        # Independently construct the publisher's variable-position readout input.
        ids = q.tokenizer.encode(text, add_special_tokens=False) + [q.tokenizer.eos_token_id]
        with q.torch.inference_mode():
            logits = q.model(input_ids=q.torch.tensor([ids]), attention_mask=q.torch.ones((1, len(ids)), dtype=q.torch.long)).logits.float()
            expected = q.torch.softmax(logits/q.temperature, dim=-1)[0, q.ai_index].item()
        assert abs(expected - q.score_text(text)) < 1e-6
        try:
            qwen_readout_inputs(q.tokenizer, q.torch, "word " * 1500, q.max_tokens)
        except ContextLimitError:
            pass
        else:
            raise AssertionError("Qwen must reject oversized input")
        print(json.dumps({"offlineVerified": True, "baselinePublisherParity": cases,
                          "multipleWindows": "passed", "qwenPublisherParity": expected, "contextGuards": "passed"}))


if __name__ == "__main__":
    main()
