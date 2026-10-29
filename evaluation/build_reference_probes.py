"""Prepare eight controlled variants of the existing synthetic baseline offline.

Writes a manual capture queue; does not submit any text to a website.
"""
import argparse
import hashlib
import json
from pathlib import Path


def build():
    path = Path(__file__).parent / 'fixtures/turnitindetector/synthetic_inputs.json'
    seed = next(x['text'] for x in json.loads(path.read_text(encoding='utf-8'))
                if x['id'] == 'synth-plain-baseline')
    sentences = seed.split('. ')
    sentences = [s.rstrip('.') + '.' for s in sentences]
    variants = [
        ('spaces', seed.replace(' ', '  '), 'Repeated spaces'),
        ('paragraphs', '\n\n'.join(sentences), 'Paragraph boundaries'),
        ('reverse', ' '.join(reversed(sentences)), 'Sentence order'),
        ('first-alone', sentences[0], 'First sentence without document context'),
        ('third-alone', sentences[2], 'Human-labeled reference sentence without context'),
        ('duplicate', seed + ' ' + sentences[0], 'Repeated sentence alignment'),
        ('append', seed + ' This additional sentence records the date and the room temperature.', 'Added context'),
        ('punctuation', seed.replace(';', ','), 'Single punctuation substitution'),
    ]
    rows = [{'id': 'probe-baseline', 'text': seed, 'purpose': 'Unchanged control'}]
    rows += [{'id': 'probe-' + name, 'text': text, 'purpose': purpose} for name, text, purpose in variants]
    rows += [{'id': 'probe-repeat-' + str(i), 'text': seed, 'purpose': 'Unchanged repeat'} for i in (1, 2)]
    for row in rows:
        row.update(group='synth-plain-baseline', provenance='AI-authored synthetic fixture; not a human control',
                   sha256=hashlib.sha256(row['text'].encode()).hexdigest(),
                   referenceStatus='pending-manual-capture')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reference-raw/phase2-probes.json'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build(), indent=2) + '\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
