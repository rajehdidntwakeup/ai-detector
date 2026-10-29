# Local engine evaluation

Run from the repository root using the existing environment:

```powershell
.\.venv\Scripts\python.exe -m evaluation.run --output evaluation/results/baseline.json
.\.venv\Scripts\python.exe -m evaluation.setup_candidates smol
.\.venv\Scripts\python.exe -m evaluation.setup_candidates qwen
.\.venv\Scripts\python.exe -m evaluation.trial_rewrite smol --output evaluation/results/smol-trial.json
.\.venv\Scripts\python.exe -m evaluation.trial_rewrite qwen --output evaluation/results/qwen-trial.json
.\.venv\Scripts\python.exe -m evaluation.recheck evaluation/results/qwen-trial.json --output evaluation/results/qwen-comparison.json
.\.venv\Scripts\python.exe -m evaluation.compare --engines baseline modernbert desklib qwen3-variable --output evaluation/results/compare-corrected-2026-10-29.json
.\.venv\Scripts\python.exe -m evaluation.calibrate --compare evaluation/results/compare-corrected-2026-10-29.json --output evaluation/results/calibrate-corrected-2026-10-29.json
```

Only setup accesses the network. Downloads are pinned by `candidates.json` and stored in ignored `models/rewrite-*` folders. Trials load safetensors locally, disable remote code, and block socket connections. They use the installed Transformers/PyTorch environment with dynamic int8 linear layers, rather than requiring a second inference runtime. Each candidate should run in a separate process so its memory is released before the next trial.

The manifest currently contains 28 screening records, including three existing AI-written regression passages and five labeled human controls. The human controls still need stronger source verification; one held-out editorial lacks an identifiable source. Only one human control is in calibration and two are in the held-out split. This is too little evidence to certify detection accuracy. The planned 120-document pilot and 30-passage rewriting review remain outstanding. Do not fill human labels with AI-written simulations of human prose.

## Corrected detector evaluation (29 October 2026)

Comparison schema 2 restores DistilBERT CLS/SEP tokens and Qwen's trained readout token. Each model runs in a fresh worker process with outbound sockets blocked. Memory fields measure OS process high-water resident memory, including native allocations; they are not incremental tensor-memory measurements. Input word counts accompany timings so short-input performance is not presented as a 400-word benchmark.

Baseline detection retains windowed full-text coverage. Candidate engines explicitly reject inputs beyond their supported context instead of silently truncating them; comparison reports keep those rows as `unsupported_length`, with no score. Calibration refuses incomplete coverage. Use an explicitly scoped, independently versioned corpus that every engine supports, or add and evaluate a long-document aggregation policy before comparing larger inputs.

Calibration verifies the manifest hash, individual text hashes, labels, splits, groups, source-code fingerprint and complete case coverage. Old schema-1 reports are rejected. Both classification tails are fitted with empirical error constraints, with valid ordered thresholds; when no human threshold is supported, lower scores remain uncertain. Missing classes or an impossible AI false-positive target prevent publication of a threshold artifact. This does not establish population error bounds on a tiny corpus and does not turn raw model scores into calibrated authorship probabilities.

Historical phase-3/phase-4 artifacts are retained for traceability but superseded for selection. No replacement detector is promoted. `localhost:3000` defaults to the corrected baseline and offers Qwen as an experimental browser option. ModernBERT and Desklib remain available through the comparison CLI.

Manifest records require `id`, `text`, `label`, `source`, `permission`, `group`, and `split`. Labels: human, ai, edited, mixed. Splits: regression, development, calibration, heldout. Choose groups conservatively by shared author/source/topic lineage; a group cannot span splits. Duplicate text and IDs are rejected. The validator cannot independently verify a claimed source or detect paraphrased duplicates.

Optional provenance fields are allowed: `language`, `domain`, `citationDensity`, `generatorFamily`, `modelVersion`, `prompt`, `editProcess`, `composition`.

Results include version identity, text hashes, per-case scores and timing, and Wilson intervals. Mixed/edited samples do not enter binary detection metrics. Small regression results must not be advertised as accuracy estimates. Inputs in the manifest are explicitly supplied evaluation material; the runner does not collect app submissions.

Rewrite trial reports contain the source and output for review. Generation stops at EOS, a token budget, or a soft time budget checked between generation steps. Non-EOS outputs are marked incomplete. Reports label review as pending: completion is not evidence of preserved meaning. Outputs must be read before any candidate is selected. Keep partial output, negation, factual changes, invented details, and instruction following as separate review dimensions.

The current trial uses three known samples to assess feasibility. It does not establish p95 latency, peak RAM, a 300-word performance gate, or quality on held-out passages. Those measurements must precede promotion to the app.

`recheck` compares original and completed candidate text with the same local detector. Incomplete outputs are skipped, and invalid outputs are reported. The research tool explicitly writes reports to the requested location; this does not change the app's memory-only text processing.

The experiment uses the project's pinned PyTorch 2.9.1 runtime. Its dynamic quantization API emits a deprecation warning; migrate to a supported runtime/quantization path before upgrading PyTorch or promoting this experiment into the application.

## Phase 1 — frozen comparison artifacts

- `baseline_manifest.json` — installed baseline identity, file checksums, thresholds, aggregation, and runtime.
- `record_baseline.py` — regenerate the manifest when the installed model changes.
- `detector_candidates.json` — detector registry (separate from rewriting candidates).
- `disagreement_schema.json` — schema for score disagreements with external services.
- `disagreements.jsonl` — current disagreement registry.
- `../.docs` — fixed protocol for candidate comparison and promotion gates.

Record the baseline manifest with:

```powershell
.\.venv\Scripts\python.exe -m evaluation.record_baseline
```

## Phase 5 — app integration artifacts

- `server.py` exposes `/api/status` with the active engine identity and capabilities: context length, verdict thresholds, method, and limitations.
- `detector.py` returns the same engine/capabilities record with every analysis result.
- `dist/app.js` and `dist/index.html` display the active engine name from `/api/status` instead of a hard-coded model caption.
- `evaluation/reference_scores.jsonl` — optional, ignored, local-only registry for future external-reference measurements keyed to exact text/engine/time. Never treated as authorship truth.
- `tests/test_server.py` verifies the active engine and capabilities are present in the status contract.

The browser offers baseline and experimental Qwen selection. Phase 4 adds explicit score semantics and full-document coverage to both results; ModernBERT and Desklib remain evaluation-only.

## turnitindetector reference capture (Phase 1 — update plan)

Artifacts for the [turnitindetector update plan](../.docs/turnitindetector-update-plan.md):

- `reference_schema.json` — capture record layout (draft-07 subset) for the reference site's behavior.
- `import_reference.py` — validates, sanitizes-scan, and cross-checks capture records: recomputes text hashes, UTF-16 unit and word counts, flags returned/submitted/truncation mismatches, rejects session-secret-looking fields, and refuses to write observed captures into committed fixture paths.
- `reference_capture_template.json` — fill-in template for a manual browser capture (DevTools or clipboard; automation is prohibited by the site's terms).
- `fixtures/turnitindetector/` — committed synthetic material only: `synthetic_inputs.json` (stable-ID boundary/alignment/composition inputs), `synthetic_responses.json` (constructed reference-shaped responses, labeled synthetic), `synthetic_captures.jsonl` (same records in capture layout). Regenerate with `python -m evaluation.build_reference_fixtures`.
- `reference_captures.jsonl` (ignored) — where observed captures land; raw material stays in ignored `reference-raw/`.

Import a capture:

```powershell
.\.venv\Scripts\python.exe -m evaluation.import_reference reference-raw\capture-01.json --output evaluation\reference_captures.jsonl
```

Observed captures are Phase 1 evidence records only. The first manual capture (`obs-2026-10-29-baseline-01`, 2026-10-29) verified one `/bff/detect` response for `synth-plain-baseline` and is paired with a local DistilBERT run in the evidence log; remaining live questions sit in `docs/turnitindetector-behavior-spec.md §8`.
# Reference-site comparison (Phase 3)

Run the existing four detector adapters plus the full baseline app against saved, sanitized website captures:

```powershell
.\.venv\Scripts\python.exe -m evaluation.reference_compare --output reference-raw/phase3-comparison.json
```

This command is offline and does not call the reference service. It reports per-engine coverage, errors, exact-text paired score differences, and common-subset metrics. Repeated texts count once in aggregate metrics. Reference percentages are not authorship labels. See [the engine decision](../.docs/turnitindetector-engine-decision.md) for measured results and outstanding data requirements.

