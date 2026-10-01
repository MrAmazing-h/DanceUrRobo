"""Create an isolated replay directory from the published robot motion."""

import argparse
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if not (a.model_dir / "robot.xml").is_file():
        p.error("Missing robot.xml; download the model first")
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "model").symlink_to(a.model_dir.resolve(), target_is_directory=True)
    shutil.copy2(repo / "examples/reference.npz", a.output / "reference.npz")


if __name__ == "__main__":
    main()
