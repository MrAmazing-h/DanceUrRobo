"""Render the audited poses with sequence-wide camera bounds, including all hand tips."""

import argparse
import json
from pathlib import Path

import imageio.v2 as iio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--bounds-run", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    z = np.load(a.run / "reference.npz")
    q = z["qpos"]
    fps = float(z["fps"])
    m = mujoco.MjModel.from_xml_path(str(a.run / "model/robot.xml"))
    d = mujoco.MjData(m)
    lower = np.full(3, np.inf)
    upper = -lower
    for row in np.concatenate([q, np.load(a.bounds_run / "reference.npz")["qpos"]]):
        d.qpos[:] = row
        mujoco.mj_forward(m, d)
        lower = np.minimum(lower, d.xpos[1:].min(0))
        upper = np.maximum(upper, d.xpos[1:].max(0))
    lower -= np.array([0.12, 0.12, 0.05])
    upper += np.array([0.12, 0.12, 0.10])
    lower[2] = min(0.0, lower[2])
    center = (lower + upper) / 2
    span = upper - lower
    tan = np.tan(np.radians(float(m.vis.global_.fovy)) / 2)
    aspect = 360 / 576
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g] != 0:
            m.geom_group[g] = 3
    renderer = mujoco.Renderer(m, height=576, width=360)
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    font = ImageFont.load_default(size=15)
    writer = iio.get_writer(
        a.output / "robot_two_views.mp4",
        fps=fps / a.stride,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    cameras = []
    for az, label, hspan in [
        (180, "DexMate 1P / F5D6", span[1]),
        (90, "Side view / zero gravity", span[0]),
    ]:
        camera = mujoco.MjvCamera()
        camera.lookat[:] = center
        camera.azimuth = az
        camera.elevation = -8
        camera.distance = max(
            3.25, 0.58 * (span[2] + 0.14 * max(span[:2])) / tan, 0.58 * hspan / (aspect * tan)
        )
        cameras.append((camera, label))
    for i in range(0, len(q), a.stride):
        d.qpos[:] = q[i]
        mujoco.mj_forward(m, d)
        images = []
        for cam, label in cameras:
            renderer.update_scene(d, camera=cam, scene_option=option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            im = Image.fromarray(renderer.render().copy())
            draw = ImageDraw.Draw(im)
            draw.text((8, 8), label, font=font, fill="white", stroke_width=1, stroke_fill="black")
            phase = float(z["source_frame_index"][i]) / 30 if "source_frame_index" in z else i / fps
            draw.text(
                (8, 550),
                f"play {i / fps:.2f}s / src {phase:.2f}s",
                font=font,
                fill="white",
                stroke_width=1,
                stroke_fill="black",
            )
            images.append(np.asarray(im))
        writer.append_data(np.concatenate(images, axis=1))
    writer.close()
    renderer.close()
    (a.output / "camera_report.json").write_text(
        json.dumps(
            {
                "bounds_min": lower.tolist(),
                "bounds_max": upper.tolist(),
                "camera_center": center.tolist(),
                "distances": [float(c.distance) for c, _ in cameras],
                "source_npz": str(a.run / "reference.npz"),
                "trajectory_changed": False,
                "shared_bounds_with": str(a.bounds_run),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
