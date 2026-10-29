"""Audited detector adapters. Each adapter honors its model's trained context and preprocessing."""
import json
import math
import os
from abc import ABC, abstractmethod
from pathlib import Path
from engines.inputs import bert_window, bounded_inputs, qwen_readout_inputs

ROOT = Path(__file__).resolve().parents[1]


class DetectorAdapter(ABC):
    """Common contract for detector inference adapters."""

    @property
    @abstractmethod
    def engine_id(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def revision(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def score_text(self, text: str) -> float:
        """Return the AI-likeness score for a single document."""
        raise NotImplementedError

    def to_engine_record(self):
        return {"id": self.engine_id, "revision": self.revision, "offline": True}


class DistilbertDetector(DetectorAdapter):
    """Existing baseline adapter; mirrors detector.Detector score_texts behavior for one document."""

    def __init__(self):
        self.model_dir = ROOT / "models" / "distilbert-detector"
        self.metadata = None
        self.tokenizer = None
        self.model = None
        self.temperature = None
        self.ai_index = None
        self.max_tokens = None
        self.torch = None

    @property
    def engine_id(self):
        return "rasbt/ai-text-detector-distilbert"

    @property
    def revision(self):
        return "66e301801e7057ec118553b8c7ba1b6abd7d2e6b"

    def load(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2) // 2)))
        self.metadata = json.loads((self.model_dir / "detector-config.json").read_text(encoding="utf-8"))
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir), local_files_only=True, trust_remote_code=False, use_safetensors=True,
        ).to("cpu").eval()
        self.temperature = float(self.metadata["temperature"])
        self.ai_index = int(self.metadata["label_mapping"]["ai"])
        self.max_tokens = min(512, int(self.metadata["max_length"]))
        if not math.isfinite(self.temperature) or self.temperature <= 0 or self.ai_index not in (0, 1):
            raise RuntimeError("Invalid baseline model metadata")
        self.score_text("A short sentence warms up the local text analysis model.")

    def _batched_score(self, ids, capacity):
        tasks = []
        for start in range(0, max(1, len(ids)), capacity):
            chunk = ids[start:start + capacity]
            tasks.append((max(1, len(chunk)), chunk))
        total = 0.0
        weight = 0
        with self.torch.inference_mode():
            for start in range(0, len(tasks), 4):
                batch = tasks[start:start + 4]
                inputs = self.tokenizer.pad(
                    [bert_window(self.tokenizer, chunk) for _, chunk in batch], padding=True, return_tensors="pt"
                )
                logits = self.model(**inputs).logits.float() / self.temperature
                probabilities = self.torch.softmax(logits, dim=-1)[:, self.ai_index].tolist()
                for (w, _), probability in zip(batch, probabilities):
                    if not math.isfinite(probability):
                        raise RuntimeError("The model returned an invalid score")
                    total += probability * w
                    weight += w
        return total / max(1, weight)

    def score_text(self, text: str) -> float:
        capacity = self.max_tokens - self.tokenizer.num_special_tokens_to_add(pair=False)
        ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
        return self._batched_score(ids, capacity)


class ModernBertDetector(DetectorAdapter):
    """ModernBERT detector with mean pooling and 8,192-token context."""

    def __init__(self):
        self.model_dir = ROOT / "models" / "detector-modernbert"
        self.metadata = None
        self.tokenizer = None
        self.model = None
        self.temperature = None
        self.ai_index = None
        self.max_tokens = None
        self.torch = None

    @property
    def engine_id(self):
        return "rasbt/ai-text-detector-modernbert"

    @property
    def revision(self):
        return "a6efd4bf6ecd852f332b2569ae0f98cae13b7985"

    def load(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2) // 2)))
        self.metadata = json.loads((self.model_dir / "detector-config.json").read_text(encoding="utf-8"))
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir), local_files_only=True, trust_remote_code=False, use_safetensors=True,
        ).to("cpu").eval()
        self.temperature = float(self.metadata["temperature"])
        self.ai_index = int(self.metadata["label_mapping"]["ai"])
        self.max_tokens = int(self.metadata["max_length"])
        if not math.isfinite(self.temperature) or self.temperature <= 0 or self.ai_index not in (0, 1):
            raise RuntimeError("Invalid ModernBERT metadata")
        self.score_text("A short sentence warms up the local text analysis model.")

    def score_text(self, text: str) -> float:
        inputs = bounded_inputs(self.tokenizer, text, self.max_tokens)
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits.float() / self.temperature
            probabilities = self.torch.softmax(logits, dim=-1)
        score = probabilities[0, self.ai_index].item()
        if not math.isfinite(score):
            raise RuntimeError("ModernBERT returned an invalid score")
        return score


class Qwen3VariableDetector(DetectorAdapter):
    """Qwen3 0.6B variable-position detector with documented readout position."""

    def __init__(self):
        self.model_dir = ROOT / "models" / "detector-qwen3-variable"
        self.metadata = None
        self.tokenizer = None
        self.model = None
        self.temperature = None
        self.ai_index = None
        self.max_tokens = None
        self.torch = None

    @property
    def engine_id(self):
        return "rasbt/ai-text-detector-qwen3-0.6b-variable"

    @property
    def revision(self):
        return "b08d92707d5de35a0b6ccf34e73461a4ab954a33"

    def load(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2) // 2)))
        self.metadata = json.loads((self.model_dir / "detector-config.json").read_text(encoding="utf-8"))
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir), local_files_only=True, trust_remote_code=False, use_safetensors=True,
        ).to("cpu").eval()
        self.temperature = float(self.metadata["temperature"])
        self.ai_index = int(self.metadata["label_mapping"]["ai"])
        self.max_tokens = int(self.metadata["max_text_length"])
        if self.metadata["readout_position"] != "variable" or self.max_tokens + 1 != self.metadata["context_length"]:
            raise RuntimeError("Unsupported Qwen readout configuration")
        if self.tokenizer.convert_tokens_to_ids(self.metadata["readout_token"]) != self.tokenizer.eos_token_id:
            raise RuntimeError("Qwen readout token does not match tokenizer EOS")
        if not math.isfinite(self.temperature) or self.temperature <= 0 or self.ai_index not in (0, 1):
            raise RuntimeError("Invalid Qwen3 metadata")
        self.score_text("A short sentence warms up the local text analysis model.")

    def count_text_tokens(self, text: str) -> int:
        """Count text tokens only; scoring appends one EOS readout token."""
        return len(self.tokenizer.encode(text, add_special_tokens=False, truncation=False))

    def score_text(self, text: str) -> float:
        inputs = qwen_readout_inputs(self.tokenizer, self.torch, text, self.max_tokens)
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits.float() / self.temperature
            probabilities = self.torch.softmax(logits, dim=-1)
        score = probabilities[0, self.ai_index].item()
        if not math.isfinite(score):
            raise RuntimeError("Qwen3 detector returned an invalid score")
        return score


class DesklibDetector(DetectorAdapter):
    """Desklib DeBERTa-based detector with mean pooling and single-logit head.

    The publisher provides a custom PreTrainedModel class in their README. We replicate
    the mean-pooling + single-logit head here with local-only weights and no remote code.
    """

    def __init__(self):
        self.model_dir = ROOT / "models" / "detector-desklib"
        self.metadata = None
        self.tokenizer = None
        self.model = None
        self.max_tokens = None
        self.torch = None

    @property
    def engine_id(self):
        return "desklib/ai-text-detector-v1.01"

    @property
    def revision(self):
        return "5fdea974cd4287c61674951ec78803aa274e2fb7"

    def load(self):
        import torch
        from transformers import AutoConfig, AutoTokenizer
        from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Model

        self.torch = torch
        torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2) // 2)))
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True, trust_remote_code=False)
        config = AutoConfig.from_pretrained(str(self.model_dir), local_files_only=True, trust_remote_code=False)
        base = DebertaV2Model._from_config(config)
        # Load base weights manually; keys are prefixed 'model.' in the safetensors file.
        from safetensors.torch import load_file
        state = load_file(str(self.model_dir / "model.safetensors"), device="cpu")
        base_state = {k[6:]: v for k, v in state.items() if k.startswith("model.")}
        base.load_state_dict(base_state, strict=True)
        base = base.to("cpu").eval()
        classifier = torch.nn.Linear(config.hidden_size, 1).to("cpu").eval()
        classifier.weight.data.copy_(state["classifier.weight"].view(1, -1))
        classifier.bias.data.copy_(state["classifier.bias"].view(-1))
        self.model = torch.nn.ModuleDict({"base": base, "classifier": classifier})
        self.max_tokens = 768
        self.score_text("A short sentence warms up the local text analysis model.")

    def score_text(self, text: str) -> float:
        inputs = bounded_inputs(self.tokenizer, text, self.max_tokens, padding=True)
        with self.torch.inference_mode():
            outputs = self.model["base"](**inputs)
            last_hidden_state = outputs[0]
            mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden_state.size()).float()
            pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            logit = self.model["classifier"](pooled)
            score = self.torch.sigmoid(logit).item()
        if not math.isfinite(score):
            raise RuntimeError("Desklib returned an invalid score")
        return score


ENGINES = {
    "baseline": DistilbertDetector,
    "modernbert": ModernBertDetector,
    "desklib": DesklibDetector,
    "qwen3-variable": Qwen3VariableDetector,
}
