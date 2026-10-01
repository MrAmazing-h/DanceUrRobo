# Third-party notices

## Robot assets

Dexmate robot descriptions and meshes: Copyright 2025 Dexmate Inc., [official dexmate-urdf](https://github.com/dexmate-ai/dexmate-urdf), Apache License 2.0. The separately downloadable research conversion retains the full upstream LICENSE and NOTICE, lists modifications, and includes mesh/conversion hashes. Changes include MuJoCo conversion, local mesh paths, a floating root, grounded placement, collision proxies/exclusions and mimic handling. This is not an official Dexmate release or endorsement. The repository's MIT license does not replace the robot asset license.

## Numerical and rendering dependencies

MuJoCo (Apache 2.0), NumPy/SciPy (BSD), OSQP where used (Apache 2.0), Pillow (HPND), ImageIO (BSD) and imageio-ffmpeg retain their respective upstream licenses. They are installed as dependencies rather than vendored. FFmpeg binary terms depend on the distributed build.

## Optional perception components

[GVHMR](https://github.com/zju3dv/GVHMR) provides the body reconstruction system; the demonstrated upstream revision is recorded in the workflow documentation. [SMPL-X](https://smpl-x.is.tue.mpg.de/) model assets require their own registration/license. [MediaPipe](https://github.com/google-ai-edge/mediapipe) provides the optional hand estimator. Their code, model weights and associated assets retain their respective terms. No learned weights, licensed body-model files, input person video or soundtrack are included here.

The bundled mapped landmarks, body rotation targets and robot trajectories are derived demonstration outputs. They are not ground-truth annotations or a new human-motion dataset. Attribution and any applicable upstream asset terms remain relevant when reusing outputs.

## Research inspiration

- [GMR — Retargeting Matters](https://arxiv.org/abs/2510.02252): task-based rest-frame and orientation alignment.
- [WARP](https://arxiv.org/abs/2606.29940): preserving shoulder-elbow-wrist structure.

The released IK/SQP is custom project code. No GMR/WARP implementation is vendored, no official reproduction is claimed, and these papers are not the baselines in the recorded before/after table.
