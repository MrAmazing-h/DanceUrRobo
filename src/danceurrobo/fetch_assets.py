"""Fetch public pinned assets with SHA-256 verification; no hardware or private paths."""

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(row, root):
    path = root / row["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha(path) != row["sha256"]:
            raise ValueError(f"Existing asset differs; refusing overwrite: {path}")
        return row["path"]
    temp = path.with_suffix(path.suffix + ".partial")
    try:
        with urllib.request.urlopen(row["url"], timeout=120) as response, temp.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        if sha(temp) != row["sha256"]:
            raise ValueError(
                f"SHA-256 mismatch: {row['path']}; incomplete file kept for inspection"
            )
        temp.replace(path)
    except Exception:
        # Preserve diagnostics; never silently accept a pointer, truncated file or mismatched model.
        raise
    return row["path"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("outputs/g1"))
    a = p.parse_args()
    root = a.output.resolve()
    manifest = json.loads(files("danceurrobo").joinpath("data/assets.json").read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(lambda row: fetch(row, root), manifest["files"]):
            print("VERIFIED", result, flush=True)
    template = files("danceurrobo").joinpath("data/g1_inspire_sonic.xml").read_bytes()
    dest = root / "models/g1_inspire_sonic.xml"
    if dest.exists() and dest.read_bytes() != template:
        raise ValueError(f"Refusing to overwrite a different model: {dest}")
    dest.write_bytes(template)
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports/assets.json").write_text(json.dumps(manifest, indent=2))
    print("Public assets verified; review upstream licenses before use.")


if __name__ == "__main__":
    main()
