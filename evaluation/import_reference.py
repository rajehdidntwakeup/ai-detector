"""Import sanitized turnitindetector.ai behavior captures into a JSONL registry.

Validates each capture record against evaluation/reference_schema.json (a small
draft-07 subset), recomputes the documented text hashes and UTF-16/word counts,
scans payloads for session secrets, and refuses to store observed captures in
committed fixture paths. Run from the repository root:

    .\\.venv\\Scripts\\python.exe -m evaluation.import_reference capture.json --output evaluation/reference_captures.jsonl
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("reference_schema.json")

# Never persisted on purpose: session identity must not enter the registry.
FORBIDDEN_SUBSTRINGS = (
    "cookie", "authorization", "xsrf", "x-csrf", "csrf", "sessionid",
    "x-api-key", "apikey", "api_key", "next-auth", "bearer",
)

# Paths that are committed by design: only constructed, clearly labeled material.
FIXTURE_MARKERS = ("fixtures/", "fixtures\\")


def load_schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


# --- minimal draft-07 subset validation ------------------------------------


def _type_matches(value, expected):
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return True


def _validate(value, schema, path, errors):
    declared = schema.get("type")
    if declared:
        expected = declared if isinstance(declared, list) else [declared]
        if not any(_type_matches(value, name) for name in expected):
            _type_name = " or ".join(expected)
            errors.append(f"{path or 'record'}: expected type {_type_name}")
            return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path or 'record'}: {value!r} is not one of {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path or 'record'}: expected constant {schema['const']!r}")
    if isinstance(value, str):
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: does not match pattern {schema['pattern']}")
        if schema.get("format") == "date-time" and not re.match(
            r"^\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?"
            r"([Zz]|[+-]\d{2}:?\d{2})$", value
        ):
            errors.append(f"{path}: not an ISO 8601 date-time")
    if isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}: missing required field {name!r}")
        for name, sub in schema.get("properties", {}).items():
            if name in value:
                _validate(value[name], sub, f"{path}.{name}" if path else name, errors)
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            _validate(item, schema["items"], f"{path}[{index}]", errors)


def validate_record(record, schema):
    errors = []
    _validate(record, schema, "", errors)
    return errors


# --- integrity and sanitization checks -------------------------------------


def sha256_utf8(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _deep_find(obj, needle, path=""):
    """Yield dotted paths whose key name mentions a forbidden substring."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{path}.{key}" if path else key
            if needle in key.lower():
                yield here
            yield from _deep_find(value, needle, here)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from _deep_find(value, needle, f"{path}[{index}]")


def scan_secrets(record):
    """Return dotted paths of any capture content that looks like session state."""
    problems = []
    for needle in FORBIDDEN_SUBSTRINGS:
        problems.extend(_deep_find(record, needle))
    return problems


def check_texts(record):
    """Cross-check hashes, counts, and truncation against the captured text.

    Recomputes documented values in place. Returns warnings: a mismatch between
    returned and submitted text is a finding to keep; a wrong recorded hash is a
    fatal problem for the caller to reject.
    """
    warnings = []
    capture_id = record.get("captureId", "?")
    request = record["request"]
    response = record["responseState"]
    submitted = request.get("submittedText")
    if isinstance(submitted, str):
        recorded = request.get("submittedTextSha256")
        actual = sha256_utf8(submitted)
        if recorded and recorded != actual:
            raise ValueError(f"{capture_id}: submittedTextSha256 does not match submittedText (expected {actual})")
        request["submittedTextSha256"] = actual
        request["submittedUtf16Units"] = len(submitted.encode("utf-16-le")) // 2
        request["submittedWordCount"] = len(submitted.split())
        editor = request.get("editorText")
        if isinstance(editor, str):
            actual = sha256_utf8(editor)
            recorded = request.get("editorTextSha256")
            if recorded and recorded != actual:
                raise ValueError(f"{capture_id}: editorTextSha256 does not match editorText")
            request["editorTextSha256"] = actual
            if submitted != editor:
                request["truncationObserved"] = editor.startswith(submitted)
                if not request["truncationObserved"]:
                    warnings.append("submitted text is neither identical to nor a prefix of the editor text")
            elif request.get("truncationObserved"):
                warnings.append("truncationObserved true but submitted equals editor text")
        elif request.get("truncationObserved"):
            warnings.append("truncationObserved set without editorText to compare against")
    returned = response.get("returnedText")
    if isinstance(returned, str):
        recorded = response.get("returnedTextSha256")
        actual = sha256_utf8(returned)
        if recorded and recorded != actual:
            raise ValueError(f"{capture_id}: returnedTextSha256 does not match returnedText (expected {actual})")
        response["returnedTextSha256"] = actual
        if isinstance(submitted, str) and returned != submitted:
            warnings.append("returned text differs from submitted text")
    return warnings


# --- registry writing and loading ------------------------------------------


def load_input(path):
    raw = Path(path).read_text(encoding="utf-8")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as error:
        # JSONL: one capture record per line. Records may also arrive pretty
        # printed inside single-line strings, so fall back per line.
        try:
            return [json.loads(line) for line in raw.splitlines() if line.strip()]
        except json.JSONDecodeError:
            raise error
    if isinstance(decoded, dict):
        return [decoded]
    if isinstance(decoded, list):
        return decoded
    raise ValueError(f"{path}: unsupported top-level type")


def _existing_ids(path):
    ids = set()
    if not Path(path).is_file():
        return ids
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(json.loads(line).get("captureId"))
    return ids


def write_registry(records, output):
    output = Path(output)
    fixture_target = "fixtures" in {part.lower() for part in output.resolve().parts}
    if fixture_target and any(r.get("observationType") != "synthetic" for r in records):
        raise ValueError("committed fixture paths accept synthetic records only; "
                         "write observed captures to the ignored local registry")
    ids = _existing_ids(output)
    lines = []
    for record in records:
        capture_id = record.get("captureId", "?")
        if capture_id in ids:
            raise ValueError(f"duplicate captureId in registry: {capture_id}")
        ids.add(capture_id)
        lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    if lines:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line + "\n")
    return output


def import_file(path, *, schema=None):
    """Validate, sanitize-scan, and cross-check every record in a capture file.

    Returns (records, report). Each record gains import metadata; each report
    row lists its schema errors, secret findings, and text warnings. Raises
    ValueError when any record is invalid or carries a secret.
    """
    schema = schema or load_schema()
    records, report = [], []
    for record in load_input(path):
        if not isinstance(record, dict):
            raise ValueError("Each capture must be a JSON object")
        problems = {
            "schema": validate_record(record, schema),
            "secrets": scan_secrets(record),
        }
        warnings = []
        if not problems["schema"] and not problems["secrets"]:
            try:
                warnings = check_texts(record)
            except ValueError as error:
                problems["schema"].append(str(error))
        record["importedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        record["importCheck"] = {"warnings": warnings, "errors": problems["schema"] + problems["secrets"]}
        records.append(record)
        report.append({
            "captureId": record.get("captureId", "?"),
            "errors": problems["schema"] + problems["secrets"],
            "warnings": warnings,
        })
    if any(row["errors"] for row in report):
        raise ValueError(f"validation failed for {path}:\n" + json.dumps(report, indent=2))
    return records, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", help="capture JSON files (single object, array, or JSONL)")
    parser.add_argument("--output", default="evaluation/reference_captures.jsonl",
                        help="ignored local registry for observed captures; fixture paths accept synthetic only")
    args = parser.parse_args(argv)
    imported = 0
    for path in args.captures:
        try:
            records, report = import_file(path)
        except (ValueError, json.JSONDecodeError) as error:
            print(f"REJECTED {path}: {error}", file=sys.stderr)
            return 1
        for row in report:
            status = "ok" if not row["warnings"] else f"warnings: {row['warnings']}"
            print(f"{row['captureId']}: {status}")
        write_registry(records, args.output)
        imported += len(records)
    print(f"imported {imported} record(s) into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
