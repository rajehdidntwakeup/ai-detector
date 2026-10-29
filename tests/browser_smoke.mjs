// Browser smoke per docs/qwen-browser-integration-plan.md step 5, step 6 of validation.
// Drives the real page in headless Chrome over the DevTools protocol:
// model selector, loading/ready states, Qwen document-level result, stale-result
// rejection, cross-tab reconciliation, accessible controls, and report export.
// Requires: server running (spawned here), Chrome, Node 22+.
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { setTimeout as sleep } from "node:timers/promises";

const ROOT = path.resolve(import.meta.dirname, "..");
const PORT = Number(process.env.BROWSER_SMOKE_PORT || 3481);
const CDP_PORT = Number(process.env.BROWSER_SMOKE_CDP || 9223);
const BASE = `http://127.0.0.1:${PORT}`;
const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const results = {};

async function waitFor(predicate, label, timeoutMs = 90000, pollMs = 250) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const value = await predicate();
    if (value) return value;
    await sleep(pollMs);
  }
  throw new Error(`Timed out waiting for: ${label}`);
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  return response.json();
}

class CdpSession {
  constructor(url) {
    this.url = url;
    this.nextId = 1;
    this.pending = new Map();
    this.listeners = new Map();
  }

  async open() {
    this.socket = new WebSocket(this.url);
    await new Promise((resolve, reject) => {
      this.socket.addEventListener("open", resolve, { once: true });
      this.socket.addEventListener("error", reject, { once: true });
    });
    this.socket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      if (message.id && this.pending.has(message.id)) {
        const { resolve, reject } = this.pending.get(message.id);
        this.pending.delete(message.id);
        if (message.error) reject(new Error(`${message.error.message}: ${message.error.data ?? ""}`));
        else resolve(message.result);
      } else if (message.method && this.listeners.has(message.method)) {
        for (const listener of this.listeners.get(message.method)) listener(message.params);
      }
    });
    return this;
  }

  on(method, listener) {
    if (!this.listeners.has(method)) this.listeners.set(method, []);
    this.listeners.get(method).push(listener);
  }

  send(method, params = {}, timeoutMs = 30000) {
    const id = this.nextId++;
    const promise = new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`CDP ${method} timed out after ${timeoutMs}ms`));
      }, timeoutMs);
      this.pending.set(id, {
        resolve: (value) => { clearTimeout(timer); resolve(value); },
        reject: (error) => { clearTimeout(timer); reject(error); },
      });
    });
    this.socket.send(JSON.stringify({ id, method, params }));
    return promise;
  }

  close() { this.socket?.close(); }
}

async function connectPage(webSocketDebuggerUrl) {
  const page = await new CdpSession(webSocketDebuggerUrl).open();
  await page.send("Runtime.enable");
  await page.send("Page.enable");
  return page;
}

async function evaluate(page, expression) {
  const response = await page.send("Runtime.evaluate", {
    expression, awaitPromise: true, returnByValue: true,
  });
  if (response.exceptionDetails) {
    throw new Error(`Page evaluation failed: ${JSON.stringify(response.exceptionDetails)}`);
  }
  return response.result.value;
}

const stage = (name) => process.stdout.write(`[browser-smoke] ${name}\n`);

const SET_TEXT = (text) => `(() => { const t = document.getElementById("writing");
  t.value = ${JSON.stringify(text)};
  t.dispatchEvent(new Event("input", { bubbles: true }));
})()`;

const WAIT_LOADED = `document.readyState === "complete" ? true
  : new Promise((resolve) => addEventListener("load", () => resolve(true), { once: true }))`;

const SET_MODEL = (name) => `(() => { const s = document.getElementById("model-select");
  s.value = ${JSON.stringify(name)};
  s.dispatchEvent(new Event("change", { bubbles: true }));
})()`;

async function analyzeAndWait(page, text) {
  await evaluate(page, SET_TEXT(text));
  await evaluate(page, `document.getElementById("analyze-button").click()`);
  return waitFor(() => evaluate(page, `!document.getElementById("result-content").hidden ? true : null`),
    "result content visible");
}

async function nextDownload(browserSession, downloadDir, renameTo) {
  // Watch the download directory instead of Browser.download* events: the
  // event ordering from headless Chrome is not reliable enough to assert on.
  const before = new Set(fs.readdirSync(downloadDir));
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    await sleep(150);
    const files = fs.readdirSync(downloadDir).filter((f) => !before.has(f) && !f.endsWith(".crdownload"));
    if (files.length > 1) throw new Error(`Unexpected extra downloads: ${files.join(", ")}`);
    if (files.length === 1) {
      const source = path.join(downloadDir, files[0]);
      let last = -1;
      let stable = 0;
      while (Date.now() < deadline) {  // wait for the write to settle
        const size = fs.statSync(source).size;
        if (size === last && (stable += 1) >= 3) break;
        if (size !== last) stable = 0;
        last = size;
        await sleep(150);
      }
      fs.renameSync(source, renameTo);
      return fs.readFileSync(renameTo, "utf8");
    }
  }
  throw new Error("Timed out waiting for a report download");
}

async function main() {
  const chromeProfile = fs.mkdtempSync(path.join(os.tmpdir(), "local-lens-chrome-"));
  const downloadDir = fs.mkdtempSync(path.join(os.tmpdir(), "local-lens-downloads-"));
  const server = spawn(path.join(ROOT, ".venv", "Scripts", "python.exe"),
    [path.join(ROOT, "server.py"), "--port", String(PORT)], { cwd: ROOT, windowsHide: true });
  const chrome = spawn(CHROME, [
    "--headless=new", `--remote-debugging-port=${CDP_PORT}`, `--user-data-dir=${chromeProfile}`,
    "--no-first-run", "--no-default-browser-check", "--disable-gpu", "--remote-allow-origins=*", "about:blank",
  ], { windowsHide: true });
  let pageA = null;
  let pageB = null;
  let browserSession = null;
  try {
    const version = await waitFor(() => fetchJson(`http://127.0.0.1:${CDP_PORT}/json/version`).catch(() => null),
      "Chrome DevTools endpoint");
    browserSession = await new CdpSession(version.webSocketDebuggerUrl).open();
    await browserSession.send("Browser.setDownloadBehavior",
      { behavior: "allow", downloadPath: downloadDir, eventsEnabled: true });

    await waitFor(async () => {
      try { return (await fetchJson(`${BASE}/api/status`)).status === "ready"; } catch { return false; }
    }, "server baseline ready", 240000, 500);

    const newTarget = (url) => fetchJson(`http://127.0.0.1:${CDP_PORT}/json/new?${url}`,
      { method: "PUT" });
    pageA = await connectPage((await newTarget(BASE)).webSocketDebuggerUrl);
    await evaluate(pageA, WAIT_LOADED);

    stage("selector");
    // --- Accessible selector with both models --------------------------------
    await waitFor(() => evaluate(pageA,
      `!document.getElementById("model-select").disabled && document.querySelectorAll("#model-select option").length === 2 ? true : null`),
      "model selector enabled with two options");
    const labels = await evaluate(pageA,
      `Array.from(document.querySelectorAll("#model-select option")).map(o => o.textContent)`);
    const labelled = await evaluate(pageA,
      `Boolean(document.querySelector('label[for="model-select"]'))`);
    results.selector = { labels, accessibleLabel: labelled };
    if (!labelled || labels.length !== 2) throw new Error("Selector accessibility or options wrong");

    stage("qwen-switch");
    // --- Switch to Qwen: loading state, then ready ----------------------------
    await evaluate(pageA, SET_MODEL("qwen3-variable"));
    const loadingText = await waitFor(() => evaluate(pageA,
      `document.getElementById("engine-status").textContent.includes("Loading Qwen3-0.6B") ? document.getElementById("engine-status").textContent : null`),
      "Loading Qwen state", 10000);
    results.loadingState = loadingText;
    await waitFor(() => evaluate(pageA,
      `document.getElementById("engine-status").textContent.includes("ready") ? true : null`),
      "Qwen ready state", 240000);
    results.readyState = await evaluate(pageA, `document.getElementById("engine-status").textContent`);

    stage("qwen-analysis");
    // --- Qwen analysis: document-level result, no fake highlights ------------
    await evaluate(pageA, SET_TEXT(
      "Clinical handover requires structured communication between ward nurses. This draft examines how documentation practices affect patient safety during shift changes."));
    await evaluate(pageA, `document.getElementById("analyze-button").click()`);
    results.analyzingState = await evaluate(pageA, `document.getElementById("engine-status").textContent`);
    await waitFor(() => evaluate(pageA, `!document.getElementById("result-content").hidden ? true : null`),
      "Qwen result visible");
    const qwenResult = await evaluate(pageA, `({
      statLabel: document.getElementById("passage-stat-label").textContent,
      statValue: document.getElementById("passage-stat").textContent,
      highlightsHidden: document.getElementById("highlights-section").hidden,
      highlightsButtonHidden: document.getElementById("highlights-button").hidden,
      saveVisible: !document.getElementById("download-button").hidden,
      score: document.getElementById("ai-score").textContent,
    })`);
    results.qwenDocumentResult = qwenResult;
    results.phase4 = await evaluate(pageA, `(() => {
      const coverage = document.getElementById("result-coverage").textContent;
      const expected = currentResult.coverage;
      const invalid = [null, NaN, "0.8", -0.1, 1.1].every(score => {
        try { validateResult({...currentResult, aiConfidence: score}); return false; } catch { return true; }
      });
      return {coverageCorrect: coverage.includes(expected.textTokensAnalyzed.toLocaleString() + " of " + expected.textTokensTotal.toLocaleString()),
              hasSeparateLimit: coverage.includes("Model limit: " + expected.textTokenLimit.toLocaleString()), invalidScoresRejected: invalid,
              humanScoreCorrect: document.getElementById("human-score").textContent === (currentResult.humanRate * 100).toFixed(2) + "%",
              modelCorrect: document.getElementById("result-model").textContent === currentResult.engine.id};
    })()`);
    if (!Object.values(results.phase4).every(Boolean)) throw new Error(`Phase 4 result details failed: ${JSON.stringify(results.phase4)}`);
    if (qwenResult.statValue !== "Document-level" || !qwenResult.highlightsHidden || qwenResult.highlightsButtonHidden !== true) {
      throw new Error(`Qwen result must be document-level without highlights: ${JSON.stringify(qwenResult)}`);
    }

    stage("stale");
    // --- Stale-result rejection: editing during analysis discards the result ---
    await evaluate(pageA, SET_TEXT("Another longer draft about nursing documentation practices follows here. ".repeat(3)));
    await evaluate(pageA, `document.getElementById("analyze-button").click()`);
    await sleep(400);
    await evaluate(pageA, SET_TEXT("Edited mid-flight, so the in-flight result must be discarded."));
    await sleep(2500);
    const stale = await evaluate(pageA, `({
      resultHidden: document.getElementById("result-content").hidden,
      emptyShown: !document.getElementById("empty-state").hidden,
    })`);
    results.staleResultDiscarded = stale.resultHidden === true && stale.emptyShown === true;
    if (!results.staleResultDiscarded) throw new Error(`Stale result was displayed: ${JSON.stringify(stale)}`);
    // Let the CPU inference finish; the status poll owns busy state.
    await waitFor(async () => (await fetchJson(`${BASE}/api/status`)).busy === false, "server idle again", 240000, 500);

    stage("cross-tab");
    // --- Cross-tab reconciliation: tab B switches, tab A follows --------------
    pageB = await connectPage((await newTarget(BASE)).webSocketDebuggerUrl);
    await evaluate(pageB, WAIT_LOADED);
    await waitFor(() => evaluate(pageB, `!document.getElementById("model-select").disabled ? true : null`),
      "tab B selector enabled", 240000);
    await evaluate(pageB, SET_MODEL("baseline"));
    await waitFor(() => evaluate(pageA, `document.getElementById("model-select").value === "baseline" ? true : null`),
      "tab A reconciled to baseline", 30000);
    results.crossTabReconciled = true;
    await waitFor(async () => (await fetchJson(`${BASE}/api/status`)).status === "ready",
      "baseline ready after cross-tab switch", 240000, 500);

    stage("baseline-analysis");
    // --- Baseline analysis still shows passage highlights ---------------------
    await analyzeAndWait(pageA,
      "The ward round starts early. Notes pile up quickly. Every entry matters for the next shift. A missed allergy can change everything. Good habits protect patients.");
    const baselineResult = await evaluate(pageA, `({
      statValue: document.getElementById("passage-stat").textContent,
      highlightsShown: !document.getElementById("highlights-section").hidden,
      passages: document.querySelectorAll(".passage").length,
      firstPassageLabel: document.querySelector(".passage")?.getAttribute("aria-label")?.slice(0, 30) ?? null,
    })`);
    results.baselineHighlights = baselineResult;
    await evaluate(pageA, `document.querySelector(".score-details").open = true`);
    await pageA.send("Emulation.setDeviceMetricsOverride", {width: 1280, height: 1000, deviceScaleFactor: 1, mobile: false});
    const layout = await pageA.send("Page.getLayoutMetrics");
    const screenshot = await pageA.send("Page.captureScreenshot", {format: "png", captureBeyondViewport: true,
      clip: {x: 0, y: 0, width: layout.cssContentSize.width, height: layout.cssContentSize.height, scale: 1}});
    fs.mkdirSync(path.join(ROOT, ".runtime"), {recursive: true});
    fs.writeFileSync(path.join(ROOT, ".runtime", "phase4-baseline.png"), Buffer.from(screenshot.data, "base64"));
    if (!baselineResult.highlightsShown || baselineResult.passages === 0 || Number(baselineResult.statValue) < 1) {
      throw new Error(`Baseline highlights broken: ${JSON.stringify(baselineResult)}`);
    }

    stage("baseline-report");
    // --- Report export: baseline report contains passage scores ----------------
    const baselineDownload = nextDownload(browserSession, downloadDir,
      path.join(downloadDir, "baseline-report.txt"));
    await evaluate(pageA, `document.getElementById("download-button").click()`);
    const baselineReport = await baselineDownload;
    results.baselineReport = {
      hasModel: baselineReport.includes("Model: rasbt/ai-text-detector-distilbert"),
      hasPassageScores: baselineReport.includes("PASSAGE SCORES"),
      hasCompleteCoverage: baselineReport.includes("Complete text") && !baselineReport.includes("undefined"),
    };

    stage("qwen-report");
    // --- Switch to Qwen again, then export a document-level report ------------
    await waitFor(() => evaluate(pageB, `!document.getElementById("model-select").disabled && document.getElementById("model-select").value === "baseline"`),
      "tab B idle before switching model", 30000);
    await evaluate(pageB, SET_MODEL("qwen3-variable"));
    await waitFor(() => evaluate(pageA, `document.getElementById("model-select").value === "qwen3-variable" && document.getElementById("result-content").hidden && currentResult === null`),
      "cross-tab model change clears prior result", 30000);
    await waitFor(() => evaluate(pageA,
      `document.getElementById("engine-status").textContent.includes("ready") && document.getElementById("model-select").value === "qwen3-variable" ? true : null`),
      "Qwen ready again", 240000);
    await analyzeAndWait(pageA,
      "Academic writing in nursing education demands careful attention to evidence and tone.");
    const qwenDownload = nextDownload(browserSession, downloadDir,
      path.join(downloadDir, "qwen-report.txt"));
    await evaluate(pageA, `document.getElementById("download-button").click()`);
    const qwenReport = await qwenDownload;
    results.qwenReport = {
      hasModel: qwenReport.includes("Model: rasbt/ai-text-detector-qwen3-0.6b-variable"),
      passageSectionOmitted: qwenReport.includes("PASSAGE SCORES: not computed (document-level model)."),
      hasCoverage: qwenReport.includes("Coverage:"),
      hasThresholdPolicy: qwenReport.includes("provisional-display-bands"),
    };
    if (!results.qwenReport.hasModel || !results.qwenReport.passageSectionOmitted
        || !results.qwenReport.hasCoverage || !results.qwenReport.hasThresholdPolicy) {
      throw new Error(`Qwen report wrong: ${JSON.stringify(results.qwenReport)}`);
    }
    if (!results.baselineReport.hasModel || !results.baselineReport.hasPassageScores || !results.baselineReport.hasCompleteCoverage) {
      throw new Error(`Baseline report wrong: ${JSON.stringify(results.baselineReport)}`);
    }

    results.browserVerified = true;
  } finally {
    pageA?.close();
    pageB?.close();
    browserSession?.close();
    chrome.kill();
    await new Promise((resolve) => chrome.once("exit", resolve)).catch(() => {});
    try {
      await fetch(`${BASE}/api/stop`, { method: "POST", headers: { "X-Local-Lens-Action": "stop" } });
    } catch { /* server may already be gone */ }
    server.kill();
    // A leftover temp profile is acceptable; a locked crashpad file must not fail the smoke.
    for (const dir of [chromeProfile, downloadDir]) {
      try { fs.rmSync(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 }); }
      catch { /* leave it to the OS temp cleanup */ }
    }
  }
  console.log(JSON.stringify(results, null, 2));
}

main().catch((error) => { console.error(error); process.exit(1); });
