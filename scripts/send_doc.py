"""Sign a PDF, post it to the webhook service, poll until it is processed, and print the result.

    python scripts/send_doc.py data/doc_002.pdf [--url http://127.0.0.1:8000] [--timeout 180]

The signature is HMAC-SHA256 over the exact bytes sent, keyed with WEBHOOK_SECRET from .env.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--timeout", type=float, default=180)
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    secret = os.environ.get("WEBHOOK_SECRET")
    if not secret:
        sys.exit("WEBHOOK_SECRET is not set in .env")
    path = Path(args.pdf)
    body = path.read_bytes()
    signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    with httpx.Client(base_url=args.url, timeout=30) as http:
        t0 = time.monotonic()
        r = http.post("/documents", content=body, headers={"Content-Type": "application/pdf",
                                                           "X-Signature": signature, "X-Filename": path.name})
        print(f"POST /documents -> {r.status_code}")
        print(json.dumps(r.json(), indent=2))
        if r.status_code not in (200, 202):
            return 1
        doc_id = r.json()["document_id"]

        result = r.json()
        while result.get("status") == "processing":
            if time.monotonic() - t0 > args.timeout:
                sys.exit(f"still processing after {args.timeout:.0f} s")
            time.sleep(1.0)
            result = http.get(f"/documents/{doc_id}").json()
        print(f"\nGET /documents/{doc_id} -> {result['status']} after {time.monotonic() - t0:.1f} s")
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
