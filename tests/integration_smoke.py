"""Real-model integration smoke: live server vs direct adapter, per docs/qwen-browser-integration-plan.md step 5.

Covers: legacy baseline behavior, Qwen token-boundary handling, Unicode and
citation-heavy input, live-vs-adapter parity within 1e-6, absence of document
files, and load/scoring time plus process memory across
baseline -> Qwen -> baseline -> Qwen. Run separately; needs downloaded models.
"""
import ctypes
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PORT = 3471
BASE = f"http://127.0.0.1:{PORT}"


class ProcessMemory:
    """Working set of the real server interpreter, without external packages.

    ``.venv\\Scripts\\python.exe`` on Windows is a launcher stub that spawns the
    base interpreter as a child; the stub stays around 4 MB, so memory is
    measured on the deepest descendant python process instead.
    """

    def __init__(self, stub_pid):
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        self._PMC = PMC
        pid = self._deepest_python_pid(stub_pid)
        self._handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not self._handle:
            raise OSError("Could not open the server process for memory queries")
        self.pid = pid

    @staticmethod
    def _deepest_python_pid(stub_pid):
        """Walk the python.exe descendant chain below the venv launcher stub.

        ``.venv\\Scripts\\python.exe`` on Windows is a launcher stub (~4 MB)
        that spawns the base interpreter as a child; the models live in the
        deepest descendant. Uses CIM via PowerShell because the Toolhelp32
        struct layout varies between Windows builds.
        """
        output = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object ProcessId,ParentProcessId | ConvertTo-Csv -NoTypeInformation"],
            capture_output=True, text=True, timeout=30)
        children = {}
        for line in output.stdout.splitlines()[1:]:
            parts = [part.strip().strip('"') for part in line.split(",")]
            if len(parts) == 2 and all(part.isdigit() for part in parts):
                children.setdefault(int(parts[1]), []).append(int(parts[0]))

        def depth(pid):
            best = 0
            for child in children.get(pid, []):
                best = max(best, depth(child) + 1)
            return best

        pid = stub_pid
        while children.get(pid):
            pid = max(children[pid], key=depth)  # follow the real interpreter chain
        return pid

    def working_set_mb(self):
        info = self._PMC()
        info.cb = ctypes.sizeof(info)
        if not ctypes.windll.psapi.GetProcessMemoryInfo(self._handle, ctypes.byref(info), info.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return info.WorkingSetSize / (1024 * 1024)

    def close(self):
        ctypes.windll.kernel32.CloseHandle(self._handle)


def request(path, payload=None, method=None, headers=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method or ("POST" if data else "GET"),
                                 headers=headers or ({"Content-Type": "application/json"} if data else {}))
    try:
        with urllib.request.urlopen(req, timeout=240) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def wait_ready(deadline_seconds=240):
    started = time.perf_counter()
    while time.perf_counter() - started < deadline_seconds:
        try:
            _, state = request("/api/status")
            if state["status"] == "ready":
                return state, time.perf_counter() - started
        except Exception:
            pass
        time.sleep(0.5)
    raise AssertionError(f"Server never became ready within {deadline_seconds}s")


def activate(name, timeout=240):
    code, payload = request("/api/model", {"model": name})
    assert code in (200, 202), (code, payload)
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        _, state = request("/api/status")
        if state["status"] == "ready" and state["activeModel"] == name:
            return time.perf_counter() - started
        if state["status"] == "error":
            raise AssertionError(f"Load failed: {state['error']}")
        time.sleep(0.5)
    raise AssertionError(f"{name} never became ready")


def snapshot_files():
    """File listing of the project tree (excluding venv/model caches and build noise)."""
    skip = {".venv", ".git", "__pycache__", "models"}
    files = set()
    for path in ROOT.rglob("*"):
        if path.is_file() and not any(part in skip for part in path.relative_to(ROOT).parts):
            files.add(str(path.relative_to(ROOT)))
    return files


def main():
    # Outbound networking is broken for the server process: any accidental
    # HTTP request must fail rather than succeed silently.
    env = dict(os.environ)
    env["HTTP_PROXY"] = env["HTTPS_PROXY"] = env["http_proxy"] = env["https_proxy"] = "http://127.0.0.1:9"
    env["NO_PROXY"] = env["no_proxy"] = ""
    # Block HF hub explicitly as well (belt and braces; detector.py also sets these).
    env["HF_HUB_OFFLINE"] = env["TRANSFORMERS_OFFLINE"] = "1"

    files_before = snapshot_files()
    server = subprocess.Popen(
        [str(ROOT / ".venv" / "Scripts" / "python.exe"), str(ROOT / "server.py"), "--port", str(PORT)],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    report = {}
    try:
        # --- Step 1: legacy baseline behavior and local restrictions -------------
        state, cold_load = wait_ready()
        report["baselineColdLoadSeconds"] = round(cold_load, 2)
        assert state["activeModel"] == "baseline" and state["requestedModel"] == "baseline"
        assert state["model"] == "rasbt/ai-text-detector-distilbert"
        assert {m["name"] for m in state["models"]} == {"baseline", "qwen3-variable"}
        for headers in ({"Host": f"example.org:{PORT}"}, {"Origin": "https://example.org"},
                        {"Sec-Fetch-Site": "cross-site"}):
            code, _ = request("/bff/detect", {"text": "Some text that stays local."}, headers=headers)
            assert code == 403, (headers, code)
        text = ("Clinical handover requires structured communication. " * 20).strip()
        code, payload = request("/bff/detect", {"text": text})
        assert code == 200 and payload["success"]
        data = payload["data"]
        assert "".join(s["text"] for s in data["segments"]) == text
        assert data["passages"] and data["engine"]["id"] == "rasbt/ai-text-detector-distilbert"
        assert data["passageScores"] is True
        assert data["resultSchemaVersion"] == 2
        assert data["coverage"]["textTokensAnalyzed"] == data["coverage"]["textTokensTotal"]
        assert not data["coverage"]["truncated"]
        assert data["scoreSemantics"]["referenceMapping"] is None
        report["baselinePassages"] = len(data["passages"])

        # Idempotent activation of the ready model returns 200 and does not reload.
        code, payload = request("/api/model", {"model": "baseline"})
        assert code == 200 and payload["status"] == "ready"

        # --- Steps 3/4: Qwen behavior, boundary, parity ---------------------------
        qwen_load = activate("qwen3-variable")
        report["qwenColdLoadSeconds"] = round(qwen_load, 2)
        _, state = request("/api/status")
        assert state["model"] == "rasbt/ai-text-detector-qwen3-0.6b-variable"
        assert state["revision"] == "b08d92707d5de35a0b6ccf34e73461a4ab954a33"

        # Load the adapter directly and score with outbound sockets denied; the
        # loopback HTTP comparisons below run with normal local access.
        import socket
        from unittest.mock import patch
        from evaluation.run import deny_network, load_manifest
        with patch.object(socket.socket, "connect", deny_network), \
             patch.object(socket.socket, "connect_ex", deny_network), \
             patch("socket.create_connection", deny_network):
            from engines import Qwen3VariableDetector
            adapter = Qwen3VariableDetector()
            adapter.load()
            regression = [r for r in load_manifest(ROOT / "evaluation/manifest.jsonl")
                          if r["split"] == "regression"][:4]
            direct_scores = {r["id"]: adapter.score_text(r["text"]) for r in regression}
            limit = adapter.max_tokens
            unit = "systematic review of nursing education "

            def build_near_limit():
                # Repeated units compress under BPE, so search for the largest
                # repeat count whose token count still fits, then go one beyond.
                count = lambda n: adapter.count_text_tokens(unit * n + "word")
                hi = 1
                while count(hi) <= limit:
                    hi *= 2
                lo = hi // 2
                while lo < hi - 1:
                    middle = (lo + hi) // 2
                    if count(middle) <= limit:
                        lo = middle
                    else:
                        hi = middle
                return unit * lo + "word"

            within = build_near_limit().strip()
            within_count = adapter.count_text_tokens(within)
            # Match the server, which validates (strips) before counting tokens.
            over = (within + " " + unit).strip()
            over_count = adapter.count_text_tokens(over)
            assert within_count <= limit, (within_count, limit)
            assert over_count > limit, (over_count, limit, within_count)
            special = {
                "unicode": "Résumés naïves 😀 — élève géniale, 检测模型, “quotes” and café-based research. " * 3,
                "citations": "Smith (2021) found that “results differ” (Smith, 2021; Jones et al., 2020, p. 14). "
                             "Cf. ISO 9001:2015 §4.2; see also https://example.org/a?b=1. " * 3,
            }
            special_counts = {label: adapter.count_text_tokens(value.strip()) for label, value in special.items()}
            special_direct = {label: adapter.score_text(value.strip()) for label, value in special.items()}
        del adapter
        import gc
        gc.collect()

        # Live server must reproduce the direct adapter within 1e-6.
        for record in regression:
            code, payload = request("/bff/detect", {"model": "qwen3-variable", "text": record["text"]})
            assert code == 200, (record["id"], payload)
            live = payload["data"]["aiConfidence"]
            assert abs(live - direct_scores[record["id"]]) < 1e-6, (record["id"], live, direct_scores[record["id"]])
            assert payload["data"]["passages"] == [] and payload["data"]["passageScores"] is False
            assert payload["data"]["coverage"]["truncated"] is False
        report["qwenParitySamples"] = len(regression)

        # Just within and just beyond the real token capacity.
        code, payload = request("/bff/detect", {"model": "qwen3-variable", "text": within})
        assert code == 200, (code, payload, within_count)
        assert payload["data"]["coverage"]["textTokensAnalyzed"] == within_count
        code, payload = request("/bff/detect", {"model": "qwen3-variable", "text": over})
        assert code == 422, (code, payload)
        assert payload["textTokens"] == over_count
        assert payload["totalTokens"] == over_count + 1
        report["qwenBoundary"] = {"within": within_count, "over": over_count, "limit": limit}

        # Unicode and citation-heavy text score without truncation.
        for label, sample in special.items():
            code, payload = request("/bff/detect", {"model": "qwen3-variable", "text": sample.strip()})
            assert code == 200, (label, code, payload)
            data = payload["data"]
            assert data["originalContent"] == sample.strip()
            assert data["coverage"]["textTokensAnalyzed"] == special_counts[label]
            assert abs(data["aiConfidence"] - special_direct[label]) < 1e-6, label
        report["unicodeAndCitations"] = "passed"

        # --- Step 6: memory and timing across baseline -> Qwen -> baseline -> Qwen -
        memory = ProcessMemory(server.pid)

        def timed_activate_with_status_latency(name):
            done = threading.Event()
            result = {}

            def run():
                try:
                    result["seconds"] = activate(name)
                except BaseException as exc:  # surfaced by the caller
                    result["error"] = exc
                finally:
                    done.set()

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            latencies = []
            while not done.wait(0.2):
                t0 = time.perf_counter()
                request("/api/status")
                latencies.append(time.perf_counter() - t0)
            if "error" in result:
                raise result["error"]
            return result["seconds"], latencies

        # Qwen is active; the plan's switch cycle is baseline -> Qwen -> baseline -> Qwen.
        load_seconds, latencies = timed_activate_with_status_latency("baseline")
        report["baselineReloadSeconds"] = round(load_seconds, 2)
        report["statusLatencyDuringLoadMsMax"] = round(max(latencies, default=0) * 1000, 1)
        report["memoryBaselineReadyMB"] = round(memory.working_set_mb(), 1)
        load_seconds, _ = timed_activate_with_status_latency("qwen3-variable")
        report["qwenReloadSeconds"] = round(load_seconds, 2)
        report["memoryQwenReadyMB"] = round(memory.working_set_mb(), 1)
        memory.close()

        # Warm scoring time for a mid-length document.
        warm = ("The systematic review examined how student nurses integrate feedback "
                "from multiple supervisors during placement rotations. " * 20).strip()
        started = time.perf_counter()
        code, payload = request("/bff/detect", {"model": "qwen3-variable", "text": warm})
        assert code == 200, payload
        report["qwenWarmScoringSeconds"] = round(time.perf_counter() - started, 2)
        report["qwenWarmDurationMs"] = payload["data"]["durationMs"]

        # --- Step 4: no document files were created -------------------------------
        created = snapshot_files() - files_before
        created = {f for f in created if not f.endswith(".pyc")}
        assert not created, f"Integration created unexpected files: {sorted(created)}"
        report["documentFilesCreated"] = sorted(created)
    finally:
        try:
            request("/api/stop", {}, headers={"X-Local-Lens-Action": "stop"})
        except Exception:
            pass
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()

    print(json.dumps({"integrationVerified": True, **report}, indent=2))


if __name__ == "__main__":
    main()
