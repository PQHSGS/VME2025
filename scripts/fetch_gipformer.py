"""Download gipformer-65M int8 ONNX files into models/gipformer-65M-i8/.

Run: python scripts/fetch_gipformer.py
Idempotent: skips files already present with correct size.
"""

import sys
import time
import urllib.request
from pathlib import Path

REPO = "g-group-ai-lab/gipformer-65M-rnnt"
FILES = ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
BASE = Path(__file__).resolve().parent.parent
DEST = BASE / "models" / "gipformer-65M-i8"


def fetch(name: str) -> None:
    url = f"https://huggingface.co/{REPO}/resolve/main/{name}"
    out = DEST / name
    tmp = out.with_suffix(out.suffix + ".part")
    started = time.perf_counter()
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("content-length", 0))
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
    tmp.rename(out)
    mb = done / 1e6
    print(f"{name}: {mb:.1f}MB in {time.perf_counter() - started:.0f}s", flush=True)


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    rc = 0
    for name in FILES:
        target = DEST / name
        if target.exists() and target.stat().st_size > 0:
            print(f"{name}: already present ({target.stat().st_size / 1e6:.1f}MB)")
            continue
        try:
            fetch(name)
        except Exception as exc:
            print(f"{name}: FAILED {exc}", file=sys.stderr, flush=True)
            target.unlink(missing_ok=True)
            target.with_suffix(target.suffix + ".part").unlink(missing_ok=True)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
