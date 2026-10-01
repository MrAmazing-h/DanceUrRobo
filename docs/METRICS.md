# Metric definitions

All source-aligned pose metrics are evaluated at the original 30 Hz timestamps, even for a 120 Hz exported robot trajectory. The frozen target files remain unchanged while optimization runs. The evaluation source is an estimated, robot-proportioned reconstruction, not motion-capture ground truth.

- **Wrist position error:** Euclidean distance between robot hand-base FK and the corresponding adapted wrist target, reported in cm across both hands and frames.
- **Arm direction error:** angle between normalized shoulder-to-elbow or elbow-to-wrist vectors, in degrees; each side/segment is reported separately.
- **Palm orientation:** construct X from pinky MCP to index MCP, orthogonalize wrist-to-middle MCP as Y, set Z to their cross product; report SO(3) geodesic angle between source and robot frames.
- **Chest orientation:** SO(3) geodesic angle between the anatomical source chest frame and robot `torso_l3` rotation. Chest tilt is separately the angle of its up-axis to world vertical.
- **Head orientation:** compose the SMPL-X parent chain, align the first source head orientation to the first anatomical chest frame, and compare robot head orientation with that calibrated target.
- **Inter-wrist vector:** norm of the difference between left-minus-right wrist vectors, in cm.
- **Acceleration / jerk:** absolute second/third finite differences of scalar hinge coordinates at common 30 Hz, multiplied by 30²/30³. Summaries aggregate coordinates and frames; these are not vector norms, torque estimates or contact metrics.
- **Collision:** maximum penetration by pair class and counts of samples exceeding the declared tolerance. The public audit includes stored poses and ground checks as well as intermediate samples. The original recorded report only counted intermediate samples and recorded ground depth separately.

Each error has mean, 95th percentile and maximum. A low average does not certify individual frames. `results/before.json` and `after.json` preserve the complete original reports. The baseline and final trajectory share targets and robot geometry, but several objectives changed together; this is not an isolated objective ablation.

The public audit also rejects nonfinite arrays, invalid root quaternions, joint-position limit violations and mimic residuals above 1e-6 rad. These state checks do not enforce actuator speed or torque.

No success rate, balance metric, finger-contact accuracy or hardware execution score is claimed. No statistical generalization is established from one sequence.
