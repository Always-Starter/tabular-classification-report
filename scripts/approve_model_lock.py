#!/usr/bin/env python3
"""Record an actual human approval of an exact lock digest; never infer consent."""
import argparse
from pathlib import Path
from common import read_json, sha, utcnow, write_json


def approve(lock_path, expected_sha256, approver, statement):
    path = Path(lock_path)
    if sha(path) != expected_sha256:
        raise ValueError("Lock digest differs from the reviewed digest")
    if read_json(path)["status"] != "pending_human_approval":
        raise ValueError("Unexpected lock status")
    if not approver.strip() or not statement.strip():
        raise ValueError("Record who approved and their explicit approval statement")
    record = {"status": "approved", "lock_sha256": expected_sha256, "approved_at": utcnow(),
              "approver": approver, "statement": statement}
    write_json(path.with_name(path.name + ".approval.json"), record, exclusive=True)
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lock", required=True, type=Path)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--approver", required=True)
    p.add_argument("--statement", required=True)
    a = p.parse_args()
    approve(a.lock, a.expected_sha256, a.approver, a.statement)
