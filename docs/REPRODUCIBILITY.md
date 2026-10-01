# Reproduction and data contract

## Validated boundaries

The primary reproducible entry is the frozen retargeting input bundle, not raw monocular video. `scripts/reproduce.py` runs the numerical refinement path from that bundle. `scripts/prepare_example.py` is faster and independently evaluates/renders the recorded final trajectory. Optional upstream adapters are supplied for new videos, but upstream model installation, licensed assets and visual review remain separate steps.

The model archive contains the exact converted robot, collision proxies, mimic equations, upstream license, conversion provenance and per-file hashes. Keep its relative paths intact. Both packages reject a different joint ordering even if it also has 55 qpos coordinates. The archive SHA-256 is embedded in `fetch_model.py`.

## Frozen input layout

| File | Required arrays |
| :--- | :--- |
| `seed.npz` | `qpos[T,55]`, scalar `fps=30` |
| `target_inputs.npz` | `human_joint_motions[T,12,3]`, `joint_names[12]`, matching `mapping_keys` and `mapping_links`, `fps=30` |
| `body_targets.npz` | `chest_rotation[T,3,3]`, `head_rotation[T,3,3]` |
| `hands.npz` | `valid[T,2]` boolean observation flags |
| `model/` | `robot.xml`, `robot.urdf`, `meshes/` (local symlink or directory) |

The release includes 590 frames. Units: metres, radians, seconds; world Z is up and initial forward is approximately +X. Root quaternions are wxyz. `examples/motion_manifest.json` records joint names, addresses and the model digest beside the portable motions. No SMPL-X body model, shape coefficients, raw video or soundtrack is bundled. Only the mapped target landmarks and derived body orientations needed for the optimizer are distributed. These remain estimates.

The recorded final `examples/reference.npz` has 2,360 frames at 120 Hz. `hand_observed` is true only at source-aligned observed samples; inserted samples are not counted as additional observations. `source_frame_index` refers to 30 Hz source time. The final source pose is held during the last quarter-frame samples to preserve clip duration.

## Historical selection and portable reproduction

The selected path was iteration01 → iteration04 → iteration05 → iteration06 → 120 Hz interpolation → local clearance correction. Iteration02 was a separate filtering/temporal experiment. Iteration03 explored velocity constraints and was not selected. Their parameter records are included; they should not be mistaken for a six-stage serial pipeline.

Original execution used Python 3.11, MuJoCo 3.10.0, NumPy 2.3.5, SciPy 1.17.1, OSQP 1.1.3. The recipe uses the recorded numerical settings, but sparse solvers, BLAS and line-search contacts can cause non-bitwise-identical results on another platform. Auditing may identify different intervals and therefore invoke different local corrections. Inspect the final metrics and collision report; do not judge success from an optimizer completion message.

The public implementation changes path handling and import structure, uses derived chest/head targets in place of raw SMPL-X parameters, validates model ordering, fails explicitly on bad SQP status, and strengthens the audit to include endpoint and ground checks. It retains the selected experiment’s objective equations and its QP-bounded framewise update (rather than a later post-step clipping edit). `results/release_validation.json` documents checks actually run on this release.

New-video preparation reproduces the geometric target adapter but defaults to a neutral seed; the historical seed came from a separate earlier Laplacian solver. This initialization difference is explicit and can change convergence. Use `--seed` to provide an already validated robot trajectory when available.

## Runtime and supported systems

Framewise IK and temporal refinement use CPU numerical solvers. GVHMR reconstruction requires its upstream GPU environment; replay and evaluation do not. Rendering requires OpenGL (EGL, OSMesa or a desktop context). Linux release checks and Python 3.11/3.12 CI cover the stated scope; Windows and macOS are unvalidated. Do not run `python -O`: legacy upstream adapters still contain assertion-based checks.
