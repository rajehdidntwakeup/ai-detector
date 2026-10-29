"""Generate AI-labeled text from pinned local models for corpus expansion."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import time
from unittest.mock import patch

from evaluation.run import deny_network

ROOT = Path(__file__).resolve().parents[1]


PROMPTS = {
    "nursing-education": "Write a 200-word academic paragraph for nursing students explaining how to assess patient hydration status using physical signs and clinical judgement.",
    "health-policy": "Write a 180-word formal summary of the trade-offs between centralized and decentralized health-care administration, aimed at a public-health audience.",
    "research-methods": "Write a 220-word academic methods paragraph describing how a hospital could design a retrospective chart-review study on readmission rates.",
    "general-academic": "Write a 160-word academic paragraph about the role of peer review in maintaining research quality.",
}


def generate(candidate, prompt, max_new_tokens=400, seconds=120):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    spec = json.loads(Path(__file__).with_name("candidates.json").read_text(encoding="utf-8"))[candidate]
    path = ROOT / "models" / ("rewrite-" + candidate)
    with patch.object(socket.socket, "connect", deny_network), patch.object(socket.socket, "connect_ex", deny_network), patch("socket.create_connection", deny_network):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        torch.set_num_threads(4)
        torch.manual_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True, trust_remote_code=False, use_safetensors=True, dtype=torch.float32).eval()
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        messages = [{"role": "user", "content": prompt}]
        chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(chat, return_tensors="pt")
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, max_time=seconds,
                                       do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                                       pad_token_id=tokenizer.eos_token_id)
        duration = time.perf_counter() - started
        tokens = generated[0][inputs["input_ids"].shape[-1]:]
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        complete = bool(len(tokens) and int(tokens[-1]) in eos)
        text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        return {
            "engine": spec, "prompt": prompt, "text": text, "complete": complete,
            "generatedTokens": len(tokens), "durationSeconds": round(duration, 3),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "quantization": "torch dynamic int8 linear layers; float32 embeddings",
            "transformers": importlib.metadata.version("transformers"),
            "torch": torch.__version__,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", choices=("qwen", "smol"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=400)
    args = parser.parse_args()
    rows = []
    for key, prompt in PROMPTS.items():
        rows.append(generate(args.candidate, prompt, args.max_tokens, args.seconds) | {"topic": key})
        print(json.dumps({"candidate": args.candidate, "topic": key, "words": len(rows[-1]["text"].split()), "complete": rows[-1]["complete"]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"candidate": args.candidate, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "samples": len(rows)}))


if __name__ == "__main__":
    main()
