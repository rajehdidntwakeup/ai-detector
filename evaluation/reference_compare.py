"""Offline paired comparison of saved website captures and pinned local engines.

python -m evaluation.reference_compare --output reference-raw/phase3-comparison.json
Reference percentages are targets for agreement only, never authorship labels.
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
import subprocess
import sys
import time
from unittest.mock import patch

from evaluation.compare import ROOT, deny_network, run_engine, implementation_hash
from evaluation.import_reference import import_file
from evaluation.memory import peak_rss_bytes
from evaluation.reference_probe import number, display_percent, summarize
from engines import ENGINES


def prepare(records):
    """Deduplicate exact texts; retain all observations separately for repeats."""
    unique, observations, excluded = {}, [], []
    for record in records:
        score = record['responseState'].get('fields', {}).get('aiConfidence')
        if record['observationType'] != 'observed' or not number(score) or not 0 <= score <= 1:
            excluded.append(record['captureId'])
            continue
        text = record['request']['submittedText']
        key = hashlib.sha256(text.encode('utf-8')).hexdigest()
        unique[key] = {'id': key, 'text': text, 'label': 'unknown', 'split': 'discovery', 'group': key}
        observations.append({'captureId': record['captureId'], 'sha256': key, 'score': score})
    return list(unique.values()), observations, excluded


def ranks(values):
    return [sum(x < v for x in values) + (sum(x == v for x in values) + 1)/2 for v in values]


def correlation(a, b):
    if len(a) < 3:
        return None
    a, b = ranks(a), ranks(b)
    am, bm = statistics.mean(a), statistics.mean(b)
    numerator = sum((x-am)*(y-bm) for x,y in zip(a,b))
    denominator = math.sqrt(sum((x-am)**2 for x in a)*sum((y-bm)**2 for y in b))
    return numerator/denominator if denominator else None


def metrics(pairs):
    if not pairs:
        return {'n': 0, 'maePoints': None, 'medianAbsolutePoints': None, 'p95AbsolutePoints': None,
                'biasPoints': None, 'rankCorrelation': None, 'binaryVerdictAgreement': None}
    errors = [100*(local-reference) for local, reference in pairs]
    absolute = sorted(abs(x) for x in errors)
    return {'n': len(pairs), 'maePoints': statistics.mean(absolute),
            'medianAbsolutePoints': statistics.median(absolute),
            'p95AbsolutePoints': absolute[math.ceil(.95*len(absolute))-1],
            'biasPoints': statistics.mean(errors),
            'rankCorrelation': correlation([p[0] for p in pairs], [p[1] for p in pairs]),
            'binaryVerdictAgreement': statistics.mean((a >= .5) == (b >= .5) for a,b in pairs)}


def compare(results, observations, total):
    references = {}
    for observation in observations:
        references.setdefault(observation['sha256'], []).append(observation['score'])
    # Exact repeats have equal document weight rather than inflating the sample.
    means = {key: statistics.mean(values) for key, values in references.items()}
    valid = [{row['sha256']: row['score'] for row in result.get('rows', [])
              if row['status'] == 'ok' and number(row['score']) and 0 <= row['score'] <= 1}
             for result in results]
    common = set(means)
    for scores in valid:
        common.intersection_update(scores)
    output = []
    for result, scores in zip(results, valid):
        paired = sorted(set(scores) & set(means))
        output.append({'name': result['name'], 'eligibleDocuments': total, 'scoredDocuments': len(scores),
                       'coverage': len(scores)/total if total else None,
                       'allSupported': metrics([(scores[k], means[k]) for k in paired]),
                       'commonSupported': metrics([(scores[k], means[k]) for k in sorted(common)]),
                       'failedOrUnsupported': total-len(scores)})
    return {'commonDocuments': len(common), 'engines': output,
            'verdictRule': 'Diagnostic binary AI score >= 0.5 for BOTH sides; not website three-class verdict or app uncertainty bands.',
            'uncertainty': 'Intervals unavailable: independent source lineage and sufficient groups have not been established.',
            'referenceAggregation': 'Mean of observed raw AI scores per exact submitted-text hash; repeat ranges reported separately.'}


def worker(name, cases):
    if name != 'baseline-app':
        return run_engine(name, cases)
    from detector import Detector
    start = time.perf_counter()
    engine = Detector()
    load_ms = round(1000*(time.perf_counter()-start))
    rows, identity = [], None
    for case in cases:
        start = time.perf_counter()
        try:
            result = engine.analyze(case['text'])
            identity = result['engine']
            if result['originalContent'] != case['text']:
                raise ValueError('App normalization changed submitted text; excluded from exact-text comparison')
            row = {'status': 'ok', 'score': result['aiConfidence'],
                   'passageCount': len(result['passages']), 'truncated': False}
        except (ValueError, RuntimeError) as exc:
            row = {'status': 'error', 'score': None, 'message': str(exc)}
        rows.append({'id': case['id'], 'sha256': case['id'],
                     'durationMs': round(1000*(time.perf_counter()-start)), **row})
    return {'engine': identity, 'loadMs': load_ms, 'peakProcessRssBytes': peak_rss_bytes(), 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, default=Path('evaluation/reference_captures.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=[*ENGINES, 'baseline-app'])
    args = parser.parse_args()
    records, _ = import_file(args.captures)
    cases, observations, excluded = prepare(records)
    if not cases:
        parser.error('No observed captures with finite AI scores in [0,1]')
    if args.worker:
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
        with patch.object(socket.socket, 'connect', deny_network), patch.object(socket.socket, 'connect_ex', deny_network), patch('socket.create_connection', deny_network):
            result = worker(args.worker, cases)
        print(json.dumps(result, allow_nan=False))
        return
    results = []
    for name in ['baseline', 'modernbert', 'desklib', 'qwen3-variable', 'baseline-app']:
        print(f'Running {name}', flush=True)
        try:
            process = subprocess.run([sys.executable, '-m', 'evaluation.reference_compare', '--worker', name,
                                      '--captures', str(args.captures.resolve()), '--output', str(args.output.resolve())],
                                     cwd=ROOT, capture_output=True, text=True, timeout=1800)
            if process.returncode:
                raise RuntimeError(process.stderr[-2000:])
            result = json.loads(process.stdout)
        except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError) as exc:
            result = {'rows': [], 'error': str(exc)}
        results.append({'name': name, **result})
        print(f'Finished {name}', flush=True)
    digest = hashlib.sha256()
    for name in ['evaluation/reference_compare.py', 'evaluation/import_reference.py', 'evaluation/reference_probe.py']:
        digest.update((ROOT/name).read_bytes())
    report = {'schemaVersion': 1, 'createdAt': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'captureFileSha256': hashlib.sha256(args.captures.read_bytes()).hexdigest(),
              'inferenceImplementationSha256': implementation_hash(), 'comparisonImplementationSha256': digest.hexdigest(),
              'python': platform.python_version(), 'torch': importlib.metadata.version('torch'),
              'transformers': importlib.metadata.version('transformers'),
              'networkBlockedDuringInference': True, 'results': results, 'observations': observations,
              'excludedCaptures': excluded, 'agreement': compare(results, observations, len(cases)),
              'referenceBehavior': summarize(records),
              'authorshipQuality': {'status': 'not_evaluated', 'reason': 'Reference scores are not authorship labels. Use the separate provenance corpus.'},
              'decision': 'retain_baseline_pending_independent_reference_and_authorship_evaluation',
              'limitations': ['Discovery only; no mapping fitted, no default changed.',
                              'One fresh process per engine; single-pass scoring includes first-use effects, not warmed p95.',
                              'Adapter results are document-only; baseline-app includes all passage inference.',
                              'No fresh confirmation set or validated source-group confidence intervals.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report['agreement'], indent=2))


if __name__ == '__main__':
    main()
