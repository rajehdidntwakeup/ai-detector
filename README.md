# Local Lens

A local AI writing detector at **http://localhost:3000**. The interface, server, and downloaded model all run on your laptop. No API key, subscription, GPU, or internet connection is needed after setup.

## Use it

On this laptop, dependencies and the model have been installed in this project.

1. Double-click **`start.cmd`** in this folder.
2. Wait for “Local model ready.” The browser opens automatically.
3. Paste English text (up to 2,000 words), pick a detector above the editor, and choose **Analyze text**.
4. Double-click **`stop.cmd`** to stop the app. You can also press Ctrl+C in its server window.

Keep the server running while using the page. Running `start.cmd` again opens the existing instance. If another application occupies port 3000, run `start.cmd --port 3001` and use the address it prints; stop it with `stop.cmd --port 3001`.

For a fresh installation, install Python 3.10–3.13 and run **`setup.cmd`** once with an internet connection. Setup downloads a CPU-only PyTorch runtime and the pinned models: the baseline DistilBERT detector (about 270 MB) and the experimental Qwen3-0.6B detector (about 1.2 GB). Allow roughly 3 GB of disk space for the environment and models. Later launches work offline.

From PowerShell:

```powershell
.\.venv\Scripts\python.exe server.py --open
```

## What the result means

Open **Score details** after analysis to see the human class score, complete-document token coverage, and the model used. Token capacity is shown separately for Qwen. Reports include the same coverage information. Uncertain remains a score band, not a mixed-authorship percentage; no reference-website score mapping is applied. See the [Phase 4 integration report](.docs/turnitindetector-phase4-integration.md).

Two detectors are available above the editor. The default is [rasbt/ai-text-detector-distilbert](https://huggingface.co/rasbt/ai-text-detector-distilbert); **Qwen3-0.6B Detector (experimental)** is [rasbt/ai-text-detector-qwen3-0.6b-variable](https://huggingface.co/rasbt/ai-text-detector-qwen3-0.6b-variable). Both are fine-tuned English-language classifiers licensed under Apache-2.0. This is an independent implementation inspired by the reference workflow; it does not run the reference website's backend or reproduce Turnitin's scores.

- The **AI model score** is the model's AI-class output, with its published temperature scaling. It is not a validated real-world probability of authorship.
- **DistilBERT:** documents longer than 512 model tokens are analyzed in non-overlapping windows. The document score is the token-weighted mean of their predictions; this aggregation has not been independently calibrated. Passages are scored independently, with up to 80 highlight units. Very sentence-dense text combines adjacent sentences. Long passages also use token windows. Short excerpts may be unreliable, and their scores need not agree with the document score.
- **Qwen3-0.6B (experimental):** the whole document is scored in a single pass. It accepts at most 1,023 text tokens (about 700–800 English words); longer input is rejected with the exact token count and is never truncated or windowed. No passage highlights are computed; the result is document-level only. Scoring can take tens of seconds on a laptop CPU.
- Display thresholds are human-leaning at or below 35%, uncertain between 35% and 65%, and AI-leaning at or above 65%. For the experimental model these bands are provisional. “Uncertain” does not mean mixed authorship; the binary classifier has no independently learned mixed class.
- Text over 2,000 words or 50,000 characters is rejected with a message; nothing is silently truncated. At least 10 characters and some letters are required. Aim for 80+ English words.
- Models can misclassify human writing and miss AI writing, especially on new generators, edited text, short excerpts, and languages or domains outside their training data. No local accuracy benchmark or equivalence to the reference detector is claimed. The two models can disagree strongly on the same text; the score alone does not settle which is right.

## Privacy and offline behavior

- The server binds to **127.0.0.1**, so it is not exposed on your Wi-Fi network.
- Analysis uses local safetensors weights with `local_files_only=True`, `trust_remote_code=False`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`.
- The page has no analytics, CDNs, external fonts, or external assets. A Content Security Policy restricts connections to the same local origin.
- Submitted text and results stay in browser/server memory. The app does not write them to a database, logs, browser storage, or model cache. Clearing the editor removes the current UI result. Closing the app ends its process; this is not a claim of secure memory erasure or control over OS/browser memory handling.
- **Open .txt** reads a local file in your browser. **Save report** explicitly downloads a text report to your chosen download location.
- Only `setup.cmd` / `setup_model.py` contact package/model hosts. They do not read or upload user documents.

## Implementation

`dist/` is a buildless HTML/CSS/JavaScript UI. `server.py` serves local assets and `/bff/detect` (also available as `/api/analyze`). `detection_service.py` owns the allowlisted model registry (`baseline`, `qwen3-variable`), one-model-at-a-time loading with a shared operation lock, and the Qwen document-level result. `detector.py` owns text validation, windowed CPU inference, passage segmentation, and exact text reconstruction. `engines/` holds the audited per-model adapters. `setup_model.py` downloads the pinned model revisions, including model cards and licenses, into ignored `models/`.

To check status: `GET /api/status`. To activate a model: `POST /api/model` with `{"model":"qwen3-variable"}` (202 while loading, 200 if ready). To analyze text: `POST /bff/detect`, content type `application/json`, body `{"text":"Your text here...","model":"baseline"}`; omitting `model` means the baseline. Requests must target localhost/127.0.0.1; cross-site browser requests are rejected. Only one inference or model load runs at a time; competing requests receive 429. A request for a model that is not active receives 409 rather than another model's score. Qwen text over 1,023 text tokens receives 422 with the exact counts. The response includes model identity/revision, scores, coverage, passages (baseline only), exact text segments, duration, and notices. No simulated scores are returned on failure.

## Verification

The [Phase 5 acceptance review](.docs/turnitindetector-phase5-acceptance.md) retains DistilBERT as default. It passes the measured laptop workloads and local regression checks; Qwen remains experimental because of speed and context limits. Reference-score equivalence and broad detection accuracy remain unproven.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tests\offline_smoke.py
.\.venv\Scripts\python.exe tests\engine_parity_smoke.py
.\.venv\Scripts\python.exe tests\integration_smoke.py
node tests\browser_smoke.mjs
node --check dist\app.js
```

The offline smoke test disables outbound socket connections before importing/loading the model, then runs actual inference and checks result invariants. It verifies offline operation, not classification accuracy.

The engine parity smoke test requires the downloaded Qwen detector. It checks baseline and Qwen scores against their documented token construction, long-window reconstruction, and explicit context-limit rejection. These checks validate inference correctness, not authorship accuracy.

The integration smoke test starts the real server with a broken proxy (so any accidental outbound request fails), then checks legacy baseline behavior, live-server-vs-direct-adapter parity within 1e-6, Qwen token-boundary rejection with exact counts, Unicode and citation-heavy input, load and scoring times, process memory across baseline → Qwen → baseline → Qwen, and that no document files are created. The browser smoke test drives the real page in headless Chrome: model selection, loading/analyzing states, document-level Qwen results, stale-result rejection, cross-tab reconciliation, and report export. See [docs/qwen-integration-readiness.md](.docs/qwen-integration-readiness.md) for the recorded results.

The separate rewriting roadmap is [docs/undetectable-integration-plan.md](.docs/undetectable-integration-plan.md): research Undetectable.ai's observable behavior and extend Local Lens with rewriting, comparison, and rechecking while keeping local processing as the default. This is a plan; rewriting is not implemented yet.

The current requested roadmap is the [turnitindetector.ai update plan](.docs/turnitindetector-update-plan.md), refreshed on 29 October 2026 against the site's public frontend and the working app. It starts with real response captures and paired comparisons, then specifies measured local updates. The [academic detector improvement plan](.docs/detector-improvement-plan.md) remains the companion protocol for labeled data, calibration, and default-model quality; its earlier Undetectable.ai discrepancy is separate from this target.

Implementation has started with the [offline evaluation tools](evaluation/README.md): a provenance-checked detector baseline and pinned local rewriting feasibility trials. These research commands are separate from the app; rewriting is not yet enabled in the interface.

See the [measured results](.docs/engine-evaluation.md) and [provisional engine decision](.docs/engine-decision.md). The initial trial rejected SmolLM2's tested configuration and kept Qwen as experimental pending broader preservation tests.

The original detector research is preserved in [docs/reverse-engineering-plan.md](.docs/reverse-engineering-plan.md). Local implementation deliberately uses Python for inference and a buildless interface instead of adding a Node framework. Grammar, citation, humanizer, and plagiarism tools remain outside this first version.

## Sources and model provenance

- Baseline: [model card and limitations](https://huggingface.co/rasbt/ai-text-detector-distilbert), [pinned inference metadata](https://huggingface.co/rasbt/ai-text-detector-distilbert/blob/66e301801e7057ec118553b8c7ba1b6abd7d2e6b/detector-config.json), revision `66e301801e7057ec118553b8c7ba1b6abd7d2e6b`; local copy in `models/distilbert-detector/`.
- Qwen (experimental): [model card](https://huggingface.co/rasbt/ai-text-detector-qwen3-0.6b-variable), revision `b08d92707d5de35a0b6ccf34e73461a4ab954a33`; local copy in `models/detector-qwen3-variable/`.
