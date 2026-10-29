"""Record the Phase 5 evidence review without changing the app or model selection.

Requires the phase5 benchmark, unit and integration logs in .runtime and retained
Phase 4 browser evidence. Run: python -m evaluation.record_acceptance
"""
import hashlib
from collections import Counter
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def performance(report, gate):
    document = report['summary']['document']
    full = report['summary']['full-analysis']
    doc_ok = (document['supportedCases'] == document['eligibleCases']
              and document['p95Seconds'] is not None and document['p95Seconds'] <= gate['maxWarmDocumentP95Seconds'])
    full_ok = (full['supportedCases'] == full['eligibleCases']
               and full.get('maxSeconds') is not None and full['maxSeconds'] <= gate['maxFullAnalysisSeconds'])
    ram_ok = report['peakProcessRssBytes'] < gate['maxPeakProcessRSSBytes']
    return {'status': 'passed_measured_workloads' if doc_ok and full_ok and ram_ok else 'failed_default_workload_gate',
            'documentSpeedPass': doc_ok, 'fullCoverageAndSpeedPass': full_ok, 'memoryPass': ram_ok,
            'measurements': report['summary'], 'peakProcessRssBytes': report['peakProcessRssBytes'],
            'initializationSeconds': report['initializationSeconds']}


def main():
    protocol = read_json('evaluation/acceptance_protocol.json')
    baseline = read_json('.runtime/phase5-baseline.json')
    qwen = read_json('.runtime/phase5-qwen.json')
    unit = (ROOT / '.runtime/phase5-unit-tests.log').read_text(encoding='utf-8')
    integration = (ROOT / '.runtime/phase5-integration.log').read_text(encoding='utf-8')
    browser = (ROOT / '.runtime/phase4-browser.log').read_text(encoding='utf-8')
    count = re.search(r'Ran (\d+) tests', unit)
    unit_ok = count is not None and unit.strip().endswith('OK')
    integration_ok = '"integrationVerified": true' in integration
    browser_ok = '"browserVerified": true' in browser
    manifest = [json.loads(line) for line in (ROOT/'evaluation/manifest.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    captures = [json.loads(line) for line in (ROOT/'evaluation/reference_captures.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    observed = [r for r in captures if r.get('observationType') == 'observed']
    files = ['evaluation/acceptance_protocol.json', '.runtime/phase5-baseline.json', '.runtime/phase5-qwen.json',
             '.runtime/phase5-unit-tests.log', '.runtime/phase5-integration.log', '.runtime/phase4-browser.log',
             'docs/turnitindetector-phase4-integration.md', 'evaluation/baseline_manifest.json',
             'evaluation/selection.json', 'evaluation/results/selection-before-reference-phase5.json',
             'evaluation/manifest.jsonl', 'reference-raw/phase3-comparison.json']
    source = ['detector.py', 'detection_service.py', 'model_info.py', 'server.py', 'dist/index.html', 'dist/app.js',
              'dist/style.css', 'engines/__init__.py', 'engines/inputs.py', 'evaluation/benchmark_acceptance.py',
              'evaluation/record_acceptance.py', 'tests/integration_smoke.py']
    hashes = {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files + source}
    report = {
        'schemaVersion': 1, 'protocol': protocol['protocol'],
        'reviewedAt': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'decision': 'retain_baseline_no_promotion',
        'baselinePerformance': performance(baseline, protocol['performance']),
        'qwenPerformance': performance(qwen, protocol['performance']),
        'localRegression': {'status': 'passed' if unit_ok and integration_ok and browser_ok else 'failed',
                            'unitTests': int(count[1]) if count else None, 'unitPassed': unit_ok,
                            'liveIntegrationPassed': integration_ok, 'retainedBrowserEvidencePassed': browser_ok,
                            'offlineAndPreprocessingParity': 'Phase 4 real-model passes retained; runtime app sources unchanged in Phase 5'},
        'referenceFidelity': {'status': 'unproven', 'independentConfirmationDocuments': 0, 'recordedObservedCaptures': len(observed),
                              'reason': 'Only one inspected reference discovery document; missing runtime/UI observations and declared local differences.'},
        'referenceApproximation': {'status': 'unproven', 'reason': 'No fresh confirmation set, validated mapping, or paired improvement interval.'},
        'authorshipQuality': {'status': 'unproven', 'records': len(manifest), 'labelCounts': dict(Counter(r['label'] for r in manifest)),
                             'reason': 'Human controls have incomplete provenance; no independent promotion evaluation.'},
        'selectionUnchanged': hashes['evaluation/selection.json'] == hashes['evaluation/results/selection-before-reference-phase5.json'],
        'baselineManifestUnchanged': hashes['evaluation/baseline_manifest.json'] == 'f8b602409128eaa5b676afafc24f292c04e9e2ba485cfbeee230054f68dafdce',
        'artifactHashes': hashes,
        'limits': ['Acceptance review is not a claim that all gates passed.',
                   'Performance covers three synthetic workload cases per size with limited repeats.',
                   'Test-log markers summarize recorded runs; hashes preserve the exact evidence files.',
                   'Browser and offline/parity evidence is retained from Phase 4, not a fresh rerun.']}
    path = ROOT / 'evaluation/results/reference-phase5-acceptance.json'
    path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'artifactHashes'}, indent=2))


if __name__ == '__main__':
    main()
