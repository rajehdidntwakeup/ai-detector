"""Analyze imported reference observations offline; optionally score with our baseline.

python -m evaluation.reference_probe --local --output reference-raw/phase2-report.json
No reference-service requests are made. Synthetic records are excluded.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import socket
from unittest.mock import patch

from evaluation.import_reference import import_file


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def display_percent(value):
    if not number(value):
        return None
    return math.floor(min(1, max(0, value)) * 100 + 0.5)


def analyze(record):
    state = record['responseState']
    submitted = record['request']['submittedText']
    returned = state.get('returnedText')
    fields = state.get('fields', {})
    score, human = fields.get('aiConfidence'), fields.get('humanRate')
    ai_display, human_display = display_percent(score), display_percent(human)
    sentences = state.get('sentences', [])
    cursor, aligned, covered, values, weights = 0, 0, 0, [], []
    sums = []
    for sentence in sentences:
        text = sentence['text']
        pos = returned.find(text, cursor) if isinstance(returned, str) and text else -1
        if pos >= 0:
            aligned += 1
            covered += len(text)
            cursor = pos + len(text)
        probs = sentence.get('classProbabilities') or {}
        value = probs.get('ai')
        if number(value):
            values.append(value)
            weights.append(len(text.split()))
        parts = [probs.get(key) for key in ('ai', 'human', 'mixed')]
        sums.append(sum(parts) if all(number(x) for x in parts) else None)
    hypotheses = {}
    if values and len(values) == len(sentences) and sum(weights) and number(score):
        for name, prediction in {
            'sentenceMean': sum(values) / len(values),
            'wordWeightedMean': sum(v*w for v, w in zip(values, weights)) / sum(weights),
            'wordShareAt065': sum(w for v, w in zip(values, weights) if v >= .65) / sum(weights),
        }.items():
            hypotheses[name] = {'prediction': prediction, 'residualPoints': 100*(score-prediction)}
    badges = fields.get('aiRateResult')
    return {
        'captureId': record['captureId'], 'capturedAt': record['capturedAt'],
        'submittedSha256': hashlib.sha256(submitted.encode()).hexdigest(),
        'referenceScore': score, 'humanScore': human,
        'scoreSum': score + human if number(score) and number(human) else None,
        'computedDisplay': {'ai': ai_display, 'human': human_display,
                            'mixed': max(0, 100-ai_display-human_display)
                            if ai_display is not None and human_display is not None else None},
        'observedDisplay': record.get('display'),
        'returnedEqualsSubmitted': returned == submitted if returned is not None else None,
        'editorEqualsSubmitted': record['request'].get('editorText') == submitted
            if record['request'].get('editorText') is not None else None,
        'sentenceCount': len(sentences), 'alignedSentences': aligned,
        'coveredCharacters': covered,
        'returnedCharacters': len(returned) if isinstance(returned, str) else None,
        'sentenceProbabilitySums': sums, 'aggregationHypotheses': hypotheses,
        'outsideSentenceRange': not min(values) <= score <= max(values)
            if values and len(values) == len(sentences) and number(score) else None,
        'outOfRangeBadges': {k:v for k,v in badges.items() if number(v) and not 0 <= v <= 1}
            if isinstance(badges, dict) else {},
        'limitations': record.get('unresolved', []),
    }


def summarize(records, detector=None):
    rows, groups = [], defaultdict(list)
    for record in records:
        if record['observationType'] != 'observed':
            continue
        row = analyze(record)
        if detector is not None:
            try:
                local = detector.analyze(record['request']['submittedText'])
                row['local'] = {'score': local['aiConfidence'], 'engine': local['engine'],
                                'durationMs': local['durationMs'],
                                'scoredTextEqualsSubmitted': local['originalContent'] == record['request']['submittedText']}
                if number(row['referenceScore']) and row['local']['scoredTextEqualsSubmitted']:
                    row['local']['differencePoints'] = 100*(local['aiConfidence']-row['referenceScore'])
            except (ValueError, RuntimeError) as exc:
                row['local'] = {'error': str(exc)}
        rows.append(row)
        if number(row['referenceScore']):
            groups[row['submittedSha256']].append(row['referenceScore'])
    return {'schemaVersion': 1, 'rows': rows,
            'repeatability': [{'submittedSha256': key, 'count': len(scores),
                               'rangePoints': 100*(max(scores)-min(scores)) if len(scores)>1 else None}
                              for key, scores in groups.items()],
            'notice': 'Offline analysis of previously recorded observations. Computed displays are not observed UI values. Repeats may span unknown backend versions.'}


def deny_network(*args, **kwargs):
    raise RuntimeError('Network access is disabled during reference analysis')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, default=Path('evaluation/reference_captures.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()
    records, _ = import_file(args.captures)
    with patch.object(socket.socket, 'connect', deny_network), patch.object(socket.socket, 'connect_ex', deny_network), patch('socket.create_connection', deny_network):
        detector = None
        if args.local:
            from detector import Detector
            detector = Detector()
        report = summarize(records, detector)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'observations': len(report['rows'])}))


if __name__ == '__main__':
    main()
