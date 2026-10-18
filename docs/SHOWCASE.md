# One dance. Two robots.

The homepage uses one unified, source-time-aligned video: human SMPL-X reference, G1 with Inspire hands, and DexMate Vega-1P/F5D6 in front and side views.

This is a presentation of existing results, not a new optimization, training run or physical comparison benchmark.

## Timing and provenance

- The G1 human input reproduces all 590 chest and head rotation frames in the frozen DexMate targets exactly. The build script checks both arrays, not just matching durations.
- The output covers the common source interval [0, 19.6) seconds at native speed: 588 frames at 30 Hz.
- DexMate frame i is used at source time i/30. G1 video time is source time + 2 seconds, removing its initial standing period.
- The 25 Hz G1 video uses nearest-frame selection on the 30 Hz composite timeline. Some frames repeat; no optical-flow interpolation, state interpolation or speed change is applied. The maximum sampling-time difference is recorded in the validation JSON.
- G1's final hold and the small unmatched tail are excluded. Original full-length recordings remain available.
- Human visualization is root-centered. Video crops remove the baked captions and empty border; model geometry and trajectories are not changed.

The G1 panel is a **gravity-enabled dynamic rollout**; DexMate is a **zero-gravity kinematic replay**. Those labels remain visible throughout. Neither result is certified for hardware deployment.

## Rebuild

After the native G1 run in the main README:

```bash
mkdir -p outputs/showcase_inputs
gh release download v0.1.0 --repo MrAmazing-h/DanceUrRobo \
  --pattern dexmate_dance.mp4 --dir outputs/showcase_inputs
python scripts/build_showcase.py \
  --g1 outputs/g1/outputs/dance_native/dynamic_rollout.mp4 \
  --dexmate outputs/showcase_inputs/dexmate_dance.mp4 \
  --output outputs/showcase_new
```

Use a compute allocation for encoding on shared clusters. The output directory must be new. The script produces a 1600×760 H.264 MP4, a 960-pixel-wide 10 Hz looping GIF, review stills and a provenance/validation JSON. Input and output SHA-256 checksums are recorded. The complete MP4 is decoded again and its frame count checked.
