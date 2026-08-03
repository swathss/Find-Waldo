from __future__ import annotations

import argparse
import json
from pathlib import Path

from waldo_ai.audit import audit_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit YOLO labels, images, class counts and split leakage.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/audit"))
    args = parser.parse_args()
    summary = audit_dataset(args.dataset, args.output)
    print(json.dumps(summary, indent=2))
    if summary["issue_count"]:
        print(f"Audit completed with {summary['issue_count']} issue(s); inspect {args.output}.")


if __name__ == "__main__":
    main()

