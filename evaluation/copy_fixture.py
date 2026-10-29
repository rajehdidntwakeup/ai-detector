"""Print or copy a synthetic reference fixture input by its stable ID.

Run from the repository root:

    .\\.venv\\Scripts\\python.exe -m evaluation.copy_fixture --list
    .\\.venv\\Scripts\\python.exe -m evaluation.copy_fixture synth-plain-baseline --clipboard

The text stays deterministic and locally constructed; piping it to the reference
site is a manual browser action by the user, never automated here.
"""
import argparse
import json
import subprocess
from pathlib import Path

FIXTURES = Path(__file__).with_name("fixtures") / "turnitindetector" / "synthetic_inputs.json"


def load_inputs():
    return {item["id"]: item for item in json.loads(FIXTURES.read_text(encoding="utf-8"))}


def copy_to_clipboard(text):
    subprocess.run(["powershell", "-NoProfile", "-Command", "Set-Clipboard"],
                   input=text, text=True, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("id", nargs="?", help="fixture input ID, e.g. synth-plain-baseline")
    parser.add_argument("--list", action="store_true", help="list IDs, UTF-16 units, word counts, and purposes")
    parser.add_argument("--clipboard", action="store_true", help="copy the text to the Windows clipboard")
    args = parser.parse_args(argv)
    inputs = load_inputs()
    if args.list or not args.id:
        for identifier, item in inputs.items():
            print(f"{identifier}: {item['utf16Units']} UTF-16 units, {item['wordCount']} words - {item['purpose']}")
        return 0
    item = inputs.get(args.id)
    if item is None:
        parser.error(f"unknown fixture ID {args.id!r}; use --list")
    text = item["text"]
    # Trailing whitespace is significant: the frontend trims, the fixture reports
    # the exact string. Copy without adding newlines.
    if args.clipboard:
        copy_to_clipboard(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())