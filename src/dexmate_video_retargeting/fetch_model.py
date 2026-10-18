"""Download the versioned, SHA-256 checked robot conversion."""

import argparse
import hashlib
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/MrAmazing-h/DanceUrRobo/releases/download/v0.1.0/vega_1p_f5d6_model.zip"
SHA256 = "5caa67cb27f8f1d36efb5daa7c9af9062b0c681a8f1cdb981558aad088439f37"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("external"))
    a = p.parse_args()
    destination = a.output / "vega_1p_f5d6"
    if destination.exists():
        p.error("Model destination already exists; use a fresh output directory")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "model.zip"
        urllib.request.urlretrieve(URL, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError("Model archive checksum mismatch")
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                name = Path(info.filename)
                if (
                    name.is_absolute()
                    or ".." in name.parts
                    or not name.parts
                    or name.parts[0] != "vega_1p_f5d6"
                ):
                    raise ValueError("Unexpected archive member")
            z.extractall(a.output)
    print(destination.resolve())


if __name__ == "__main__":
    main()
