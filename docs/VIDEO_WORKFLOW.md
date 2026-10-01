# Processing a new video

The released demo can be reproduced without this optional front end. New monocular inputs require visual checks and may fail under occlusion, camera motion, multiple people or poor lighting.

## 1. Prepare and inspect video

Use a single-person, full-body clip with both hands visible where possible. `scripts/prepare_video.py` accepts a 30 Hz input and an explicit end-frame count, strips audio, preserves timing and verifies every decoded frame. It does not silently resample arbitrary video rates.

```bash
python scripts/prepare_video.py --input input.mp4 --output outputs/video --end-frame-exclusive 590
```

Change 590 to your inspected endpoint. To use another input rate, resample deliberately with FFmpeg first and document the timing change.

## 2. GVHMR reconstruction in its own environment

Follow [official GVHMR installation](https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md), including checkpoints and separately licensed SMPL/SMPL-X assets. The original demonstration used upstream revision `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1`. No upstream code or weights are vendored here.

Set `DEMO_REPO` and `RUN_DIR` to absolute paths to this checkout and your output directory. In the activated GVHMR environment, from its repository root:

```bash
python "$DEMO_REPO/scripts/check_2d.py" --video "$RUN_DIR/video/dance_30fps_verified.mp4" --output "$RUN_DIR/keypoints"
```

Review `overlay_raw.mp4`, wrist identity, track selection and low-confidence frames. Then run GVHMR's official demo:

```bash
python tools/demo/demo.py --video "$RUN_DIR/video/dance_30fps_verified.mp4" --output_root "$RUN_DIR/gvhmr" -s
python "$DEMO_REPO/scripts/export_and_evaluate_smplx.py" --prediction "$RUN_DIR/gvhmr/dance_30fps_verified/hmr4d_results.pt" --keypoints "$RUN_DIR/keypoints/vitpose.pt" --output "$RUN_DIR/body"
python "$DEMO_REPO/scripts/export_smplx_zup.py" --source "$RUN_DIR/body/smplx_native_export.npz" --output "$RUN_DIR/body_zup"
```

The `-s` option follows the original static-camera demo setup; check its suitability for your clip against upstream documentation. These adapters consume locally generated trusted PyTorch files. Reprojection diagnostics compare to estimated 2D keypoints, not measured 3D truth. Z-up export rotates around the SMPL-X root pivot and verifies reconstructed joints numerically. GVHMR's default hand pose does not estimate finger articulation.

## 3. Prepare robot targets and solve

In this package's environment:

```bash
dexmate-prepare-targets --human "$RUN_DIR/body_zup/human_smplx_zup.npz" --model-dir external/vega_1p_f5d6 --output "$RUN_DIR/inputs"
dexmate-posture-ik --inputs "$RUN_DIR/inputs" --output "$RUN_DIR/body_ik" --torso-weight 1.5 --iterations 20
dexmate-refine --inputs "$RUN_DIR/inputs" --seed "$RUN_DIR/body_ik" --output "$RUN_DIR/refined" --iterations 18 --accel 1.5 --samples 4 --velocity-factor 0
```

The new-video adapter ports the demonstrated scaling and anatomical correspondence. Its neutral initialization differs from the historical Laplacian seed; review the result and supply `--seed` when a matching, audited robot motion exists. Targets can be unreachable despite anatomically sensible mapping. Do not overwrite target files to make evaluation appear better.

## 4. Optional visible-hand articulation

Use a separate MediaPipe environment with `mediapipe==0.10.21`, compatible OpenCV and NumPy 1.x (the legacy `mp.solutions.hands` API is required). This environment need not import MuJoCo or this package. The 2D body keypoints are NumPy arrays.

```bash
python scripts/track_hands.py --video "$RUN_DIR/video/dance_30fps_verified.mp4" --body-2d "$RUN_DIR/keypoints/keypoints_2d.npz" --output "$RUN_DIR/hands_initial"
python scripts/refine_hands_crops.py --video "$RUN_DIR/video/dance_30fps_verified.mp4" --body-2d "$RUN_DIR/keypoints/keypoints_2d.npz" --initial "$RUN_DIR/hands_initial/hands.npz" --output "$RUN_DIR/hands_crops"
```

Review hand identity/visibility. Back in the core environment, first audit the body motion, then fit fingers:

```bash
dexmate-audit --run "$RUN_DIR/refined" --output "$RUN_DIR/refined/evaluation/interpolation_audit.json" --substeps 8
dexmate-fit-hands --body-run "$RUN_DIR/refined" --hands "$RUN_DIR/hands_crops/hands.npz" --output "$RUN_DIR/with_hands"
```

If the body audit fails, repair the body before running the hand fitter. The fitter expects an already usable body baseline. Long occlusions cannot be solved uniquely by interpolation. This release retains observation masks and reports contracted articulation.

## 5. Export, audit, inspect

```bash
dexmate-upsample --input "$RUN_DIR/with_hands" --output "$RUN_DIR/dense120"
dexmate-audit --run "$RUN_DIR/dense120" --output "$RUN_DIR/dense120/evaluation/interpolation_audit.json" --substeps 64
dexmate-quality --inputs "$RUN_DIR/inputs" --run "$RUN_DIR/dense120"
MUJOCO_GL=egl dexmate-render --run "$RUN_DIR/dense120" --bounds-run "$RUN_DIR/dense120" --output "$RUN_DIR/dense120/videos"
```

Without optional hand fitting, substitute `refined` for `with_hands`. The audit returns nonzero on tolerance violations. Inspect its details, optionally try `dexmate-polish --run ... --output ...` for small local corrections, then repeat the audit and metrics on that corrected output. Failure to find a small correction is an explicit failure, not a reason to hide contacts.

This sequence produces a kinematic NPZ and a video. Physical execution requires separate actuator, wheel, contact, stability and safety validation.
