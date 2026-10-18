"""Rendering recorded dynamics; never relabel kinematic data as a rollout."""

import json

import mujoco
import numpy as np


def render_saved(m, arr, out, title):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    renderer = mujoco.Renderer(m, height=480, width=640)
    d = mujoco.MjData(m)
    camera = mujoco.MjvCamera()
    camera.distance = 2.7
    camera.azimuth = 130
    camera.elevation = -15
    count = 0
    with imageio.get_writer(
        str(out / "dynamic_rollout.mp4"), fps=25, codec="libx264", quality=7
    ) as writer:
        for i in range(0, len(arr["time"]), 2):
            d.qpos[:] = arr["qpos"][i]
            mujoco.mj_forward(m, d)
            camera.lookat[:] = [d.qpos[0], d.qpos[1], 0.75]
            renderer.update_scene(d, camera)
            pic = Image.fromarray(renderer.render())
            ImageDraw.Draw(pic).text(
                (10, 10),
                f"{title} | recorded DYNAMIC rollout | gravity -9.81\nt={arr['time'][i]:.2f}s | no tether / no root override",
                fill="white",
                stroke_width=1,
                stroke_fill="black",
            )
            writer.append_data(np.asarray(pic))
            count += 1
    renderer.close()
    (out / "video.json").write_text(json.dumps(dict(frames=count, fps=25, duration=count / 25)))
