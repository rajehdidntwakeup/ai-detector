"use strict";

const $ = (id) => document.getElementById(id);
const writing = $("writing");
const modelSelect = $("model-select");
let engineReady = false;
let engineBusy = false;
let analyzing = false;
let currentResult = null;
let request = null;
let requestNumber = 0;
let elapsedTimer = null;
let sampleLoaded = false;
let modelChoices = [];
let selectedModel = "baseline";
let serverStatus = "loading";
let serverError = null;

const sample = "Last Saturday, I tried to fix the little shelf above my desk. It had been leaning to the left for months, and I was tired of catching the books before they slid off. I borrowed a drill from my neighbor and watched a tutorial twice. The first hole was in the wrong place. So was the second. By lunchtime there was dust everywhere, the cat had stolen a screw, and the shelf was still sitting on the floor. I finally got it level after measuring the brackets properly. It isn't a spectacular achievement, but seeing my books stay put makes me oddly happy.";

function modelInfo(name) {
  return modelChoices.find((model) => model.name === name) || null;
}

function modelLabel(name) {
  return modelInfo(name)?.label ?? "the selected model";
}

function words(text) {
  return text.trim() ? text.trim().split(/\s+/u).length : 0;
}

function inputError() {
  if (words(writing.value) > 2000) return "Over 2,000 words. Shorten your draft; nothing will be cut automatically.";
  if ([...writing.value.trim()].length > 50000) return "Over 50,000 characters. Please shorten your draft.";
  return "";
}

function switchingBlocked() {
  return analyzing || serverStatus === "loading" || engineBusy;
}

function updateControls() {
  const count = words(writing.value);
  const error = inputError();
  $("word-count").textContent = count.toLocaleString();
  $("word-count").parentElement.classList.toggle("over", Boolean(error));
  $("clear-button").disabled = !writing.value && !currentResult;
  modelSelect.disabled = switchingBlocked() || modelChoices.length === 0;
  $("analyze-button").disabled = !engineReady || engineBusy || analyzing || writing.value.trim().length < 10 || Boolean(error);
  $("analyze-label").textContent = analyzing ? "Analyzing…" : serverStatus === "loading" ? "Loading model…" : engineBusy ? "Model is busy…" : "Analyze text";
  if (error) {
    $("input-help").textContent = error;
    $("input-help").classList.add("warning");
  } else {
    $("input-help").classList.remove("warning");
    $("input-help").textContent = sampleLoaded ? "AI-written demo · The model can misclassify it. Try your own text." : "English text · Up to 2,000 words per check";
  }
}

function hideError() {
  $("error").hidden = true;
  $("error").textContent = "";
}

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = false;
}

function resetResult() {
  requestNumber += 1;
  request?.abort();
  request = null;
  if (analyzing) engineBusy = true; // CPU inference can finish after the HTTP request is cancelled.
  analyzing = false;
  clearInterval(elapsedTimer);
  currentResult = null;
  $("result-card").setAttribute("aria-busy", "false");
  $("empty-state").hidden = false;
  $("loading-state").hidden = true;
  $("result-content").hidden = true;
  $("highlights-section").hidden = true;
  $("highlighted-text").replaceChildren();
  $("passage-detail").textContent = "Select a passage for its score";
  $("result-tag").textContent = "READY WHEN YOU ARE";
  hideError();
  updateControls();
}

function renderModelChoices() {
  const previous = selectedModel;
  modelSelect.replaceChildren(...modelChoices.map((model) => new Option(model.label, model.name)));
  if (modelChoices.some((model) => model.name === previous)) modelSelect.value = previous;
  const info = modelInfo(selectedModel);
  $("model-note").textContent = info?.experimental
    ? `Experimental · up to ${info.textTokenLimit?.toLocaleString() ?? "its limit"} text tokens`
    : info?.passageScores ? "Passage highlights" : "";
}

function applyServerState(state) {
  engineBusy = Boolean(state.busy);
  serverStatus = state.status;
  serverError = state.error || null;
  if (Array.isArray(state.models) && state.models.length) {
    modelChoices = state.models;
    const known = new Set(modelChoices.map((model) => model.name));
    // Reconcile the selector with the server (another tab may have switched),
    // without asking the server to switch anywhere.
    const serverModel = (state.requestedModel && known.has(state.requestedModel) && state.requestedModel)
      || (state.activeModel && known.has(state.activeModel) && state.activeModel)
      || (known.has(selectedModel) && selectedModel) || "baseline";
    selectedModel = serverModel;
    if (currentResult && currentResult.engine?.id !== modelInfo(serverModel)?.engineId) resetResult();
    renderModelChoices();
  }
  const label = modelLabel(selectedModel);
  engineReady = state.status === "ready" && state.activeModel === selectedModel;
  const engineName = state.engine?.id ?? state.model ?? "Local model";
  $("engine-name").textContent = engineName.split("/").pop();
  if (analyzing) return; // Keep "Analyzing with …" until the check finishes.
  if (state.status === "loading") {
    $("engine-dot").className = "status-dot loading";
    $("engine-status").textContent = `Loading ${label} on your laptop…`;
  } else if (state.status === "error") {
    $("engine-dot").className = "status-dot offline";
    $("engine-status").textContent = "Local model needs setup";
    showError(serverError || "Run setup.cmd, then restart the app.");
  } else if (engineReady) {
    $("engine-dot").className = "status-dot";
    $("engine-status").textContent = `${label} ready · Offline mode`;
  } else {
    $("engine-dot").className = "status-dot loading";
    $("engine-status").textContent = `Waiting for ${label}…`;
  }
}

async function checkEngine() {
  try {
    const response = await fetch("/api/status", { cache: "no-store", signal: AbortSignal.timeout(5000) });
    if (!response.ok) throw new Error("Local app is unavailable.");
    const state = await response.json();
    if (state.app !== "local-lens") throw new Error("Unexpected server.");
    applyServerState(state);
  } catch {
    engineReady = false;
    serverStatus = "offline";
    $("engine-dot").className = "status-dot offline";
    $("engine-status").textContent = "Disconnected · Run start.cmd to reconnect";
  }
  updateControls();
  setTimeout(checkEngine, engineReady && !engineBusy && !analyzing ? 6000 : 1500);
}

async function selectModel(name) {
  if (name === selectedModel || switchingBlocked()) return;
  const label = modelLabel(name);
  selectedModel = name;
  resetResult(); // Clear the previous result; keep the text.
  serverStatus = "loading"; // Optimistic; the next status poll confirms.
  $("engine-dot").className = "status-dot loading";
  $("engine-status").textContent = `Loading ${label} on your laptop…`;
  updateControls();
  const number = requestNumber;
  try {
    const response = await fetch("/api/model", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: name }), signal: AbortSignal.timeout(10000),
    });
    if (number !== requestNumber) return;
    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      showError(payload?.message || `Could not activate ${label}.`);
    }
  } catch {
    if (number === requestNumber) showError(`Could not activate ${label}. Check that the local app is running.`);
  }
}

function coverageDescription(coverage) {
  if (!coverage || !Number.isInteger(coverage.textTokensAnalyzed) || !Number.isInteger(coverage.textTokensTotal)) return "Unavailable";
  const count = `${coverage.textTokensAnalyzed.toLocaleString()} of ${coverage.textTokensTotal.toLocaleString()} text tokens analyzed`;
  const limit = Number.isInteger(coverage.textTokenLimit) ? ` · Model limit: ${coverage.textTokenLimit.toLocaleString()} text tokens` : "";
  return `${count}${coverage.truncated ? " · Truncated" : " · Complete text"}${limit}`;
}

function validateResult(result) {
  const validScore = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
  if (!result || !validScore(result.aiConfidence) || !validScore(result.humanRate)
      || !["ai", "human", "uncertain"].includes(result.verdict)
      || typeof result.engine?.id !== "string" || typeof result.engine?.revision !== "string"
      || typeof result.originalContent !== "string" || !Array.isArray(result.segments)
      || !result.segments.every((segment) => typeof segment.text === "string")
      || result.segments.map((segment) => segment.text).join("") !== result.originalContent
      || !Array.isArray(result.passages) || !Array.isArray(result.notices)) {
    throw new Error("The local model returned an incomplete or invalid result. Please retry.");
  }
  if (result.passageScores !== false && (!result.passages.every((passage) => validScore(passage.aiScore))
      || !result.segments.every((segment) => segment.passageIndex === undefined || validScore(segment.aiScore)))) {
    throw new Error("Passage scores are missing or invalid. Please retry.");
  }
}

function displayResult(result) {
  const documentLevel = result.passageScores === false;
  const score = Math.round(result.aiConfidence * 100);
  const copy = {
    ai: ["Leans toward AI-generated", "The model found patterns it associates with AI-generated writing. Review the highlighted passages in context."],
    human: ["Leans toward human-written", "The model found patterns it associates with human-written text. This does not establish who wrote it."],
    uncertain: ["An uncertain signal", "The model doesn't strongly favor either class. This score cannot establish whether the writing is human, AI-generated, or edited."],
  }[result.verdict];
  $("ai-score").textContent = score;
  $("human-score").textContent = `${(result.humanRate * 100).toFixed(2)}%`;
  $("result-coverage").textContent = coverageDescription(result.coverage);
  $("result-model").textContent = result.engine.id;
  $("result-model").title = `Revision: ${result.engine.revision}`;
  $("verdict").textContent = copy[0];
  $("score-explanation").textContent = documentLevel
    ? "The model scored the complete document in one pass. Read the score with your own judgment of the writing."
    : copy[1];
  $("ring-fill").setAttribute("stroke-dasharray", `${(score / 100 * 307.88).toFixed(2)} 307.88`);
  $("ring-fill").setAttribute("class", `ring-fill ${result.verdict}`);
  $("score-ring").setAttribute("aria-label", `AI model score: ${score} percent`);
  $("scale-marker").style.left = `calc(${score}% - ${score / 100 * 4}px)`;
  $("checked-words").textContent = result.wordCount.toLocaleString();
  $("passage-stat-label").textContent = documentLevel ? "Analysis" : "Passages";
  $("passage-stat").textContent = documentLevel ? "Document-level" : String(result.passages.length);
  // Share of words in AI-leaning passages; the mean score can stay low while
  // a large part of the document is flagged, the way coverage-style reports surface it.
  $("ai-portion").textContent = result.aiPortion === undefined || documentLevel
    ? "n/a"
    : `${Math.round(result.aiPortion * 100)}%`;
  $("duration").textContent = `${(result.durationMs / 1000).toFixed(1)}s`;
  $("result-notice").textContent = result.notices.join(" ");
  $("result-tag").textContent = "CHECKED ON YOUR DEVICE";
  $("empty-state").hidden = true;
  $("result-content").hidden = false;
  $("highlights-button").hidden = documentLevel;
  $("highlights-section").hidden = true;
  $("highlighted-text").replaceChildren();
  if (!documentLevel) {
    $("highlights-section").hidden = false;
    const fragment = document.createDocumentFragment();
    for (const segment of result.segments) {
      if (segment.passageIndex === undefined) {
        fragment.append(document.createTextNode(segment.text));
        continue;
      }
      const mark = document.createElement("button");
      mark.type = "button";
      mark.className = `passage ${segment.label}`;
      mark.dataset.label = segment.label;
      mark.textContent = segment.text;
      const description = `Passage ${segment.passageIndex + 1} · ${Math.round(segment.aiScore * 100)}% AI model score`;
      mark.title = description;
      mark.setAttribute("aria-label", `${description}: ${segment.text}`);
      function inspect() {
        document.querySelectorAll(".passage.selected").forEach((item) => item.classList.remove("selected"));
        mark.classList.add("selected");
        $("passage-detail").textContent = description;
      }
      mark.addEventListener("click", inspect);
      mark.addEventListener("focus", inspect);
      fragment.append(mark);
    }
    $("highlighted-text").append(fragment);
    setFilter("all");
  }
  $("engine-status").textContent = "Analysis complete · Your text stayed on this laptop";
}

function setFilter(filter) {
  document.querySelectorAll(".filter").forEach((button) => {
    const active = button.dataset.filter === filter;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  document.querySelectorAll(".passage").forEach((mark) => mark.classList.toggle("dimmed", filter !== "all" && mark.dataset.label !== filter));
}

async function analyze() {
  if ($("analyze-button").disabled) return;
  hideError();
  currentResult = null;
  const model = selectedModel;
  const expectedEngine = modelInfo(model)?.engineId ?? null;
  const label = modelLabel(model);
  const number = ++requestNumber;
  const controller = new AbortController();
  request = controller;
  analyzing = true;
  $("empty-state").hidden = true;
  $("result-content").hidden = true;
  $("highlights-section").hidden = true;
  $("loading-state").hidden = false;
  $("loading-copy").innerHTML = `Your laptop is analyzing the text with <strong>${label}</strong>.<br>This can take a little longer for longer drafts.`;
  $("result-card").setAttribute("aria-busy", "true");
  $("result-tag").textContent = "RUNNING LOCALLY";
  $("engine-dot").className = "status-dot loading";
  $("engine-status").textContent = `Analyzing with ${label}…`;
  const start = Date.now();
  $("elapsed").textContent = "0 seconds";
  elapsedTimer = setInterval(() => { $("elapsed").textContent = `${Math.floor((Date.now() - start) / 1000)} seconds`; }, 1000);
  updateControls();
  const timeout = setTimeout(() => controller.abort("timeout"), 180000);
  try {
    const response = await fetch("/bff/detect", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, text: writing.value }), signal: controller.signal,
    });
    const payload = await response.json();
    if (number !== requestNumber) return;
    if (!response.ok || !payload.success || !payload.data) throw new Error(payload.message || "Analysis failed. Please retry.");
    if (expectedEngine && payload.data.engine?.id && payload.data.engine.id !== expectedEngine) {
      throw new Error("The result did not come from the selected model and was discarded. Try again.");
    }
    validateResult(payload.data);
    displayResult(payload.data);
    currentResult = payload.data;
  } catch (error) {
    if (number !== requestNumber) return;
    currentResult = null;
    $("result-content").hidden = true;
    $("highlights-section").hidden = true;
    $("empty-state").hidden = false;
    $("result-tag").textContent = "TRY AGAIN";
    showError(controller.signal.aborted ? "The check took too long. Try a shorter passage; the local model may still be finishing this check." : error.message === "Failed to fetch" ? "Can't reach the local server. Check that start.cmd is running." : error.message);
  } finally {
    clearTimeout(timeout);
    if (number === requestNumber) {
      clearInterval(elapsedTimer);
      analyzing = false;
      request = null;
      // engineBusy stays with the status poll: an aborted browser request does
      // not cancel the CPU inference that is still running.
      $("loading-state").hidden = true;
      $("result-card").setAttribute("aria-busy", "false");
      updateControls();
    }
  }
}

writing.addEventListener("input", () => { sampleLoaded = false; resetResult(); });
writing.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    analyze();
  }
});
$("analyze-button").addEventListener("click", analyze);
$("clear-button").addEventListener("click", () => { writing.value = ""; sampleLoaded = false; resetResult(); writing.focus(); });
$("sample-button").addEventListener("click", () => {
  writing.value = sample;
  sampleLoaded = true;
  resetResult();
  writing.focus();
});
$("file-button").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  if (file.size > 300000) {
    showError("Choose a plain text file smaller than 300 KB, containing at most 2,000 words.");
  } else if (!file.name.toLowerCase().endsWith(".txt")) {
    showError("Please choose a .txt file. You can copy and paste from other documents.");
  } else {
    try {
      const text = await file.text();
      writing.value = text;
      sampleLoaded = false;
      resetResult();
      writing.focus();
    } catch { showError("That file could not be read. Try copying its text into the editor."); }
  }
  event.target.value = "";
});
modelSelect.addEventListener("change", (event) => { selectModel(event.target.value); });
$("highlights-button").addEventListener("click", () => {
  $("highlights-section").hidden = false;
  $("highlights-section").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth" });
  const firstPassage = $("highlighted-text").querySelector("button");
  firstPassage?.focus({ preventScroll: true });
});
document.querySelectorAll(".filter").forEach((button) => button.addEventListener("click", () => setFilter(button.dataset.filter)));
$("about-button").addEventListener("click", () => $("about-dialog").showModal());
$("close-about").addEventListener("click", () => $("about-dialog").close());
$("about-dialog").addEventListener("click", (event) => {
  const rect = event.currentTarget.getBoundingClientRect();
  if (event.target === event.currentTarget && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) event.currentTarget.close();
});
$("download-button").addEventListener("click", () => {
  if (!currentResult) return;
  const result = currentResult;
  const documentLevel = result.passageScores === false;
  const capabilities = result.engine?.capabilities ?? {};
  const report = ["LOCAL LENS — LOCAL ANALYSIS REPORT", new Date().toLocaleString(), "",
    `Model: ${result.engine.id}`, `Revision: ${result.engine.revision}`,
    `Method: ${result.method}`,
    "Display thresholds: at or below 35% human-leaning, at or above 65% AI-leaning"
    + (capabilities.thresholdPolicy ? ` (${capabilities.thresholdPolicy})` : " (display bands, not validated probabilities)"),
    `AI model score: ${Math.round(result.aiConfidence * 100)}%`,
    `Human class score: ${(result.humanRate * 100).toFixed(2)}%`,
    "Score meaning: binary classifier output, not a validated authorship probability. Uncertain is not a mixed class.",
    "Reference website score mapping: none.",
    `Words checked: ${result.wordCount}`, `Time taken: ${(result.durationMs / 1000).toFixed(1)}s`];
  if (!documentLevel && result.aiPortion !== undefined) {
    report.push(`AI-flagged text: ${Math.round(result.aiPortion * 100)}% of words (share of words in AI-leaning passages)`);
  }
  report.push(`Coverage: ${coverageDescription(result.coverage)}`);
  report.push("", ...result.notices, "", "TEXT", result.originalContent, "");
  if (documentLevel) {
    report.push("PASSAGE SCORES: not computed (document-level model).");
  } else {
    report.push("PASSAGE SCORES", ...result.passages.map((passage, index) => `${index + 1}. ${Math.round(passage.aiScore * 100)}% AI model score\n${passage.text}\n`));
  }
  const url = URL.createObjectURL(new Blob([report.join("\n")], { type: "text/plain;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "local-lens-report.txt";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
updateControls();
checkEngine();
