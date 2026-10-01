# Method and implementation boundaries

## Robot coordinates and correspondence

The released conversion has `nq=55`, consisting of a free-root position/quaternion plus scalar robot joints. The model contract records every joint name and qpos address. Optimization permits root x/y/yaw; root height, root roll/pitch and wheel angles remain fixed. It does not translate root motion into wheel actuation.

Three serial torso revolute joints are optimized jointly with the base, head and arms. They have parallel axes; they cannot reproduce arbitrary human torso roll. Their influence is evaluated through FK shoulder/arm landmarks and an explicit anatomical chest-frame orientation residual, rather than directly copying human spine angles. The nominal `[0.65, 1.3, 0.65]` rad configuration has cancelling signed rotations and an upright upper torso.

| Human target | Robot body | Role |
| :--- | :--- | :--- |
| Left/right shoulder | `L/R_arm_l1` | Upper-arm anchor |
| Left/right elbow | `L/R_arm_l4` | Arm bend location |
| Left/right wrist | `L/R_hand_base` | Hand root position |
| Index/middle/pinky MCP | `L/R_ff_l1`, `mf_l1`, `lf_l1` | Relative palm frame |
| Shoulder line plus pelvis-to-neck up | `torso_l3` | Chest orientation |
| Calibrated composed SMPL-X head rotation | `head_l3` | Head orientation |

The human pelvis is not mapped directly onto the chassis. Human feet and lower-body joint angles are not retargeted. Global scale follows the average shoulder-elbow-wrist chain-length ratio. Initial shoulder-center alignment sets translation; robot shoulder width and individual upper/forearm lengths replace human dimensions while preserving estimated segment directions. Palm MCP geometry uses robot-local offsets rotated by the human palm frame. These are estimated, morphology-adapted targets, not metric ground truth.

## Sequential and temporal optimization

`posture_ik.py` solves framewise differential IK with MuJoCo Jacobians and OSQP. The residual stack includes wrist/shoulder/elbow positions, normalized arm-segment directions, wrist-relative MCP geometry, anatomical chest orientation, weak nominal torso posture, and temporal regularization. Joint limits and bounded updates are applied. Infeasible linearized collision tasks may use a recorded soft-penalty fallback; a successful QP alone never constitutes a collision pass.

`batch_refine.py` optimizes the full trajectory using sparse SQP. Variables comprise planar base motion, torso/head/arms and 12 independent finger controls; mimic relations fold follower motion into the Jacobian. The objective adds first/second temporal differences and calibrated head orientation. Nearby collision distances are linearized at saved frames and intermediate collocation samples, with explicitly penalized slacks. A trust region and optional collision-merit line search stabilize updates. An unsuccessful QP raises an error in the public version rather than silently writing a complete-looking result.

The demonstrated final refinement used **no hard velocity constraint** (`velocity_factor=0`). Temporal regularization is not a guarantee of actuator speed, acceleration, torque or power feasibility. The historical velocity-constrained attempt did not become the selected trajectory; its configuration is retained for transparency.

## Hands and missing observations

GVHMR estimates the body; its default SMPL-X hand shape is not recovered finger motion. The optional hand pipeline uses MediaPipe relative 3D landmarks, assigns left/right identity by proximity to audited body wrists, and retries missing detections on wrist-guided image crops. It retains a boolean observation mask.

`fit_hands.py` fits bend-angle and normalized thumb-tip distance features to six independent controls per F5D6 hand. Follower limits restrict master ranges. Short gaps interpolate, longer gaps hold the last observation; smoothing does not turn inferred frames into observations. Collision-prone articulation is contracted toward the input body/open-hand trajectory and independently checked. A collision-invalid body baseline causes failure rather than being hidden by finger fitting.

## Export, local correction and audit

PCHIP interpolation runs on hinge coordinates and unwrapped planar yaw. The fourfold rate conversion retains original timing and holds the last source pose to complete its final frame interval. Output root quaternions are normalized analytically. Interpolated poses must be audited independently because interpolation can introduce collisions even when endpoints pass.

A small local correction search can adjust selected thumb or arm joints with a smooth sine-squared envelope, respecting mimic limits and testing the entire local transition window. In the recorded output, three corrections had 0.001 rad peak magnitude; two changed an elbow and therefore changed wrist motion. Final metrics include those corrections.

The collision audit evaluates MuJoCo proxy geometry and model exclusions. Non-seam tolerance is 0.1 mm; same-side hand/finger or distal-wrist mesh seams use 2 mm. Optimization excludes those seam pairs from clearance inequalities. Fixed exclusions in the model remain active. Public auditing checks saved poses and intermediate samples, including ground penetration. This is discrete sampling, not continuous collision detection, raw visual-mesh intersection testing or physical safety certification.

GMR motivated task-based orientation alignment; WARP motivated preserving arm structure. This code does not implement WARP's closed-form solver or establish equivalence to either paper.
