"""Re-run the demonstrated refinement sequence from frozen retargeting inputs."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    a.output.mkdir(parents=True, exist_ok=False)
    inputs = a.output / "inputs"
    shutil.copytree(repo / "examples/inputs", inputs)
    (inputs / "model").symlink_to(a.model_dir.resolve(), target_is_directory=True)

    def run(module, *args, allow_audit_failure=False):
        command = [sys.executable, "-m", "dexmate_video_retargeting." + module, *map(str, args)]
        print("RUN", *command, flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode and not (allow_audit_failure and result.returncode == 1):
            raise subprocess.CalledProcessError(result.returncode, command)

    run(
        "posture_ik",
        "--inputs",
        inputs,
        "--output",
        a.output / "iteration01",
        "--torso-weight",
        1.5,
        "--iterations",
        20,
    )
    common = [
        "--inputs",
        inputs,
        "--accel",
        1.5,
        "--velocity",
        0.08,
        "--clearance",
        0.004,
        "--palm-weight",
        0.6,
        "--velocity-factor",
        0,
        "--head-weight",
        0.3,
        "--samples",
        4,
    ]
    run(
        "batch_refine",
        *common,
        "--seed",
        a.output / "iteration01",
        "--output",
        a.output / "iteration04",
        "--iterations",
        18,
        "--torso-weight",
        1.8,
    )
    audit = a.output / "iteration04/evaluation/interpolation_audit.json"
    run(
        "audit",
        "--run",
        a.output / "iteration04",
        "--output",
        audit,
        "--substeps",
        32,
        allow_audit_failure=True,
    )
    extra = ["--line-search", "--trust", 0.06, "--audit", audit, "--dense-substeps", 32]
    run(
        "batch_refine",
        *common,
        *extra,
        "--seed",
        a.output / "iteration04",
        "--output",
        a.output / "iteration05",
        "--iterations",
        12,
        "--torso-weight",
        1.8,
    )
    run(
        "batch_refine",
        *common,
        *extra,
        "--seed",
        a.output / "iteration05",
        "--output",
        a.output / "iteration06",
        "--iterations",
        8,
        "--torso-weight",
        4.0,
    )
    dense = a.output / "dense120"
    run("upsample_review", "--input", a.output / "iteration06", "--output", dense)
    run(
        "audit",
        "--run",
        dense,
        "--output",
        dense / "evaluation/interpolation_audit.json",
        "--substeps",
        64,
        allow_audit_failure=True,
    )
    import json

    report = json.loads((dense / "evaluation/interpolation_audit.json").read_text())
    final = dense
    if report["violating_contacts"]:
        final = a.output / "polished120"
        run("polish_local_clearance", "--run", dense, "--output", final)
    run(
        "audit",
        "--run",
        final,
        "--output",
        final / "evaluation/interpolation_audit.json",
        "--substeps",
        64,
    )
    run("quality_metrics", "--inputs", inputs, "--run", final)
    print("FINAL", final, flush=True)


if __name__ == "__main__":
    main()
