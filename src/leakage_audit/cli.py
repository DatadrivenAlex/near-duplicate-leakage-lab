"""JSONL input, machine-readable JSON output, optional CI gate."""
import argparse
import json
import sys
from pathlib import Path

from leakage_audit.core import Record, audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="UTF-8 JSONL: id, split and text strings")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--mode", choices=["exact", "lsh"], default="exact")
    parser.add_argument("--permutations", type=int, default=128)
    parser.add_argument("--bands", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fail-on-match", action="store_true")
    args = parser.parse_args(argv)
    try:
        records = []
        with args.input.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or not all(
                        isinstance(row.get(k), str) for k in ("id", "split", "text")
                    ):
                        raise ValueError("id, split and text must be strings")
                    records.append(Record(row["id"], row["split"], row["text"]))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"line {line_number}: {exc}") from exc
        result = audit(records, args.threshold, args.width, args.mode, args.permutations,
                       args.bands, args.seed)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if args.fail_on_match and result["matches"] else 0


if __name__ == "__main__":
    sys.exit(main())
