"""Offline laptop acceptance workloads, in a fresh process for each app model.

python -m evaluation.benchmark_acceptance --model baseline --output .runtime/phase5-baseline.json
python -m evaluation.benchmark_acceptance --model qwen3-variable --output .runtime/phase5-qwen.json
Synthetic inputs measure performance only, not detection quality.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import time
from unittest.mock import patch

from evaluation.memory import peak_rss_bytes

TOPICS = {
    'academic': 'The review examined nursing education across several clinical settings. Students recorded feedback after each supervised shift and discussed the results with their tutors. The authors compared these observations with earlier reports while noting differences in staffing and assessment methods.',
    'general': 'I walked to the market early on Saturday and found that the usual stall had moved. The seller explained that rain had damaged the roof. We talked about the weather while she packed my apples into a paper bag.',
    'technical': 'The service receives a request and checks its input before starting a worker. Each worker records the operation status in memory. If a client disconnects the computation may continue but its result must not replace a newer request in the interface.',
}


def workload(topic, words):
    tokens = TOPICS[topic].split()
    return ' '.join((tokens * math.ceil(words / len(tokens)))[:words])


def timings(values):
    ordered = sorted(values)
    return {'count': len(values), 'p50Seconds': statistics.median(values),
            'p95Seconds': ordered[math.ceil(.95 * len(values))-1], 'maxSeconds': ordered[-1]}


def blocked(*args, **kwargs):
    raise RuntimeError('Acceptance benchmark attempted network access')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['baseline', 'qwen3-variable'], required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('At least two measured repetitions required')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    rows = []
    with patch.object(socket.socket, 'connect', blocked), patch.object(socket.socket, 'connect_ex', blocked), patch('socket.create_connection', blocked):
        from detection_service import _qwen_result, DocumentTooLongError
        from detector import Detector
        from engines import Qwen3VariableDetector
        started = time.perf_counter()
        engine = Detector() if args.model == 'baseline' else Qwen3VariableDetector()
        if args.model != 'baseline':
            engine.load()
        initialization = time.perf_counter() - started
        for words, mode in [(400, 'document'), (2000, 'full-analysis')]:
            for topic in TOPICS:
                text = workload(topic, words)
                row = {'id': f'{topic}-{words}', 'words': words, 'mode': mode,
                       'textSha256': hashlib.sha256(text.encode()).hexdigest()}

                def score():
                    if args.model != 'baseline':
                        return _qwen_result(engine, text, time.perf_counter())
                    if mode == 'document':
                        return engine.score_texts([text])[0]
                    return engine.analyze(text)

                try:
                    warmup = score()
                    durations = []
                    for _ in range(args.repeats):
                        start = time.perf_counter()
                        result = score()
                        durations.append(time.perf_counter() - start)
                        if isinstance(result, dict):
                            assert result['originalContent'] == text
                            assert result['coverage']['textTokensAnalyzed'] == result['coverage']['textTokensTotal']
                            assert not result['coverage']['truncated']
                    row.update(status='ok', seconds=durations, **timings(durations))
                    if isinstance(warmup, dict):
                        row['coverage'] = warmup['coverage']
                        row['passageCount'] = len(warmup['passages'])
                except DocumentTooLongError as exc:
                    row.update(status='unsupported_length', textTokens=exc.text_tokens,
                               textTokenLimit=exc.text_token_limit, seconds=[])
                rows.append(row)
                print(f"{args.model}: {row['id']} {row['status']}", flush=True)
    summary = {}
    for mode in ['document', 'full-analysis']:
        selected = [r for r in rows if r['mode'] == mode]
        values = [value for r in selected for value in r['seconds']]
        summary[mode] = {'eligibleCases': len(selected), 'supportedCases': sum(r['status']=='ok' for r in selected),
                         **(timings(values) if values else {'count': 0, 'p50Seconds': None, 'p95Seconds': None})}
    root = Path(__file__).resolve().parents[1]
    files = ['detector.py', 'detection_service.py', 'model_info.py', 'engines/__init__.py', 'engines/inputs.py', 'evaluation/benchmark_acceptance.py']
    report = {'schemaVersion': 1, 'model': args.model, 'createdAt': time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              'python': platform.python_version(), 'torch': importlib.metadata.version('torch'),
              'transformers': importlib.metadata.version('transformers'),
              'sourceHashes': {f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},
              'initializationSeconds': initialization, 'peakProcessRssBytes': peak_rss_bytes(),
              'outboundSocketsBlocked': True, 'warmupPerCase': 1, 'repeatsPerCase': args.repeats,
              'rows': rows, 'summary': summary,
              'limitations': ['Three repeated synthetic prose workloads, not representative accuracy data.',
                              'Initialization is in a fresh process; OS file cache is uncontrolled and baseline constructor includes a short warmup.',
                              'Warm p95 uses nearest rank over 15 measured calls by default, with no population latency guarantee.',
                              'RSS is process high-water resident memory, not system-wide usage.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'summary': summary, 'initializationSeconds': initialization,
                      'peakGiB': report['peakProcessRssBytes']/1024**3}, indent=2))


if __name__ == '__main__':
    main()
