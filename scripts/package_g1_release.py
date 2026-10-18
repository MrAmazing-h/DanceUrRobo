"""Package recorded demo artifacts without private filesystem paths or weights."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("dist/g1_release"))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    examples = Path("examples/g1")
    results = Path("results/g1")
    examples.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    model = Path("src/danceurrobo/data/g1_inspire_sonic.xml")
    audit = json.loads((a.experiment / "reports/final_audit.json").read_text())
    manifest = []
    for label, case in (("native", "full_native_v2"), ("half_speed", "full_half_v2")):
        source = a.experiment / "outputs" / case
        with np.load(source / "fullbody_motion.npz", allow_pickle=False) as z:
            fields = {k: z[k] for k in z.files}
        fields["original_model_sha256"] = fields["model_sha256"]
        fields["model_sha256"] = np.asarray(sha(model))
        fields["model"] = np.asarray("models/g1_inspire_sonic.xml")
        dest = examples / f"{label}_motion.npz"
        if dest.exists():
            with np.load(dest, allow_pickle=False) as existing:
                if set(existing.files) != set(fields):
                    raise ValueError(f"Existing export schema differs: {dest}")
                for key, value in fields.items():
                    np.testing.assert_array_equal(existing[key], value)
        else:
            np.savez_compressed(dest, **fields)
        for name in ("metrics", "review"):
            data = json.loads((source / f"{name}.json").read_text())
            (results / f"{label}_{name}.json").write_text(json.dumps(data, indent=2))
        entry = audit["results"][case]
        entry["export"] = str(dest)
        (results / f"{label}_contact_audit.json").write_text(json.dumps(entry, indent=2))
        video = a.output / f"g1_{label}_comparison.mp4"
        shutil.copy2(source / "source_vs_robot.mp4", video)
        shutil.copy2(dest, a.output / dest.name)
        manifest.append(
            {
                "motion": label,
                "video": video.name,
                "trajectory": dest.name,
                "video_sha256": sha(video),
                "trajectory_sha256": sha(dest),
                "samples": len(fields["time_s"]),
                "fps": int(fields["fps"]),
                "kind": "dynamic simulated rollout; not hardware-ready",
            }
        )
    text = json.dumps(manifest, indent=2)
    (a.output / "manifest.json").write_text(text)
    (results / "release_manifest.json").write_text(text)
    shutil.copy2("docs/G1.md", a.output / "README.md")
    shutil.copy2("docs/RESULTS.md", a.output / "RESULTS.md")
    shutil.copy2("THIRD_PARTY_NOTICES.md", a.output / "THIRD_PARTY_NOTICES.md")
    with zipfile.ZipFile(a.output / "danceurrobo_g1_review.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(a.output.iterdir()):
            if f.suffix != ".zip":
                z.write(f, f.name)
        for f in sorted(results.glob("*.json")):
            z.write(f, "results/" + f.name)
        for f in sorted(Path("licenses").glob("*.txt")):
            z.write(f, "licenses/" + f.name)
    print(text)


if __name__ == "__main__":
    main()
