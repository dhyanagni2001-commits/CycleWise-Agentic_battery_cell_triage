"""Download the raw batch files (resumable, size-checked).

  python -m cyclewise.data.download            # all batches in config (~5 GB)
"""

from __future__ import annotations

import sys
import urllib.request

from cyclewise.config import load_config, path

CHUNK = 1 << 20


def fetch(url: str, dest, expected: int | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if expected and have == expected:
        print(f"ok      {dest.name} ({have / 1e9:.2f} GB)")
        return
    if expected and have > expected:
        dest.unlink()
        have = 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        resumed = have and r.status == 206
        mode = "ab" if resumed else "wb"
        done = have if resumed else 0
        with open(dest, mode) as fh:
            while chunk := r.read(CHUNK):
                fh.write(chunk)
                done += len(chunk)
                if expected:
                    print(f"\r{dest.name}: {done / expected:6.1%}", end="", flush=True)
    print()
    size = dest.stat().st_size
    if expected and size != expected:
        raise RuntimeError(f"{dest.name}: got {size} bytes, expected {expected}; re-run to resume")


def missing() -> list[str]:
    cfg = load_config()
    return [b["file"] for b in cfg["batches"].values()
            if not path(b["file"]).exists() or (b.get("bytes") and path(b["file"]).stat().st_size != b["bytes"])]


def main() -> None:
    for b in load_config()["batches"].values():
        try:
            fetch(b["url"], path(b["file"]), b.get("bytes"))
        except Exception as e:  # noqa: BLE001 - report and continue to the next file
            print(f"\nFAILED {b['file']}: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
