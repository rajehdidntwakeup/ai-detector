from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_ID = "rasbt/ai-text-detector-distilbert"
MODEL_REVISION = "66e301801e7057ec118553b8c7ba1b6abd7d2e6b"
MODEL_DIR = ROOT / "models" / "distilbert-detector"
MODEL_FILES = (
    "config.json", "detector-config.json", "model.safetensors",
    "tokenizer.json", "tokenizer_config.json", "README.md", "LICENSE",
)
MAX_WORDS = 2000
MAX_CHARACTERS = 50000
CONTEXT_TOKENS = 512
VERDICT_THRESHOLDS = {"human": 0.35, "ai": 0.65}
SCORE_SEMANTICS = {
    "kind": "binary-classifier-score",
    "calibratedAuthorshipProbability": False,
    "mixedClass": False,
    "referenceMapping": None,
}
METHOD = ("Token-weighted mean of document windows; passage scores computed separately. "
          "AI-flagged share = fraction of words in AI-leaning passages.")
LIMITATIONS = "English-language model; scores are not proof of authorship. Short, edited, or non-native writing can be misclassified."

