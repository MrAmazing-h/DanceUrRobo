# Results and limitations

## G1 full-body experiment

The published native and half-speed videos use the corrected SONIC mode encoding. An earlier development adapter incorrectly used one-hot modes and produced nearly static motion; it is not a successful dance baseline and is not included as a release demo.

| Measurement | Native | Half speed |
| --- | ---: | ---: |
| Duration / recorded states | 24 s / 1,200 | 44 s / 2,200 |
| Video frames at 25 Hz | 600 | 1,100 |
| Maximum root tilt | 18.3397 degrees | 19.1867 degrees |
| Recorded fall / non-foot ground contact | None | None |
| Maximum sampled joint-limit excess | 0.030053 rad | 0.014033 rad |
| Minimum sampled self-contact distance | -0.012013 m | -0.006736 m |

The balance gate requires completed execution, no sampled non-foot ground contact and tilt below 35 degrees. This is an experimental diagnostic threshold, not a safety specification. Contacts and limits are audited at 50 Hz, not continuously. MuJoCo contact distance is not an independently measured exact triangle penetration depth.

**Collision-free: no. Hardware-ready: no. Exact choreography reproduction: not established.** Hand-hand collisions dominate the contact report; smaller arm/torso contacts also exist. Neutral finger targets do not remove hand geometry. Half-speed playback reduces these sampled contact/limit extrema but does not improve maximum tilt.

Bone-direction errors are recorded by segment in `results/g1/*_review.json`. Human and robot bone conventions are approximate; those diagnostics are not exact end-effector accuracy. One source sequence and two playback rates do not constitute a robustness benchmark.

Machine-readable balance, contact, timing and portable reproduction evidence is in [results/g1](../results/g1). Video and trajectory names are paired in the release manifest. Old experiment files were not overwritten; portable packaging changes path metadata, not recorded states.

The portable release was rerun using only checksum-verified public robot/controller assets. All 1,200 native-speed qpos states matched the accepted experiment exactly (maximum absolute difference 0); both newly generated 600-frame videos decoded successfully at 25 Hz. This verifies packaging parity, not additional physical safety.

## DexMate

The original results and methodology remain in [DEXMATE.md](DEXMATE.md). That track is kinematic and zero-gravity. Do not compare its completion criteria with G1 balance or describe its video as dynamically learned dancing.
