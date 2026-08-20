# scripts/hand_refine/pose_confidence.py
"""Per-frame, per-joint confidence in the Captury hand pose.

Feeds the deviation term of the optimizer: where Captury is trustworthy the refined
pose is held close to it; where the tracker clearly failed (folded-back fingers,
contracted claws, single-frame pops) the anchor is released so the physical terms can
take over. Not a binary reject - a weight, so nothing is thrown away.

Three implausibility signals, each robust (median/MAD, so a bad stretch does not set
the norm for the whole take):

  jump   angular speed of the joint's local rotation vs the take's own MAD
  range  joint angle far outside the take's own robust range (folded / contracted)
  jerk   sign-flipping acceleration, i.e. a one-frame pop rather than real motion

confidence = 1 / (1 + sum of the exceedances), so 1.0 = fully trusted.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def _mad(x, axis=0):
    med = np.median(x, axis=axis, keepdims=True)
    return np.median(np.abs(x - med), axis=axis) * 1.4826 + 1e-9


def joint_confidence(Q, joint_idx, k_jump=6.0, k_range=4.0, k_jerk=6.0,
                     floor_jump=3.0, floor_range=20.0, floor_jerk=2.0):
    """Q: (F, N, 4) local quaternions xyzw. Returns (F, J) confidence in [0,1] and parts.

    A deviation must be BOTH statistically unusual for this joint AND large in absolute
    terms (the floor_* degrees) to count. Without the floors a nearly static joint has a
    near-zero MAD, so sub-degree noise looks infinitely significant - that alone was
    firing on ~47% of samples.
    """
    F = Q.shape[0]
    J = len(joint_idx)
    conf = np.ones((F, J))
    parts = {"jump": np.zeros((F, J)), "range": np.zeros((F, J)), "jerk": np.zeros((F, J))}

    for j, k in enumerate(joint_idx):
        R = Rotation.from_quat(Q[:, k])

        # --- jump: angular speed between consecutive frames ---
        dq = np.degrees(np.linalg.norm((R[:-1].inv() * R[1:]).as_rotvec(), axis=1))
        dq = np.r_[dq[0], dq]
        scale = max(_mad(dq) * k_jump, floor_jump)
        exc_jump = np.clip((dq - np.median(dq)) / scale - 1.0, 0, None)

        # --- range: angle away from the take's own median pose ---
        ang = np.degrees(np.linalg.norm((R.mean().inv() * R).as_rotvec(), axis=1))
        scale = max(_mad(ang) * k_range, floor_range)
        exc_range = np.clip((ang - np.median(ang)) / scale - 1.0, 0, None)

        # --- jerk: second difference, catches one-frame pops ---
        jrk = np.abs(np.r_[0, np.diff(dq)])
        scale = max(_mad(jrk) * k_jerk, floor_jerk)
        exc_jerk = np.clip((jrk - np.median(jrk)) / scale - 1.0, 0, None)

        parts["jump"][:, j] = exc_jump
        parts["range"][:, j] = exc_range
        parts["jerk"][:, j] = exc_jerk
        conf[:, j] = 1.0 / (1.0 + exc_jump + exc_range + exc_jerk)

    return conf, parts


def summarize(conf, names, thr=0.5):
    lo = conf < thr
    lines = [f"低可信度样本 {lo.sum()}/{conf.size} ({lo.mean()*100:.1f}%),"
             f" 平均可信度 {conf.mean():.3f}"]
    per_joint = lo.mean(axis=0)
    worst = np.argsort(per_joint)[::-1][:5]
    for w in worst:
        if per_joint[w] > 0:
            lines.append(f"  {names[w]:32s} {per_joint[w]*100:5.1f}% 帧可信度低")
    bad_frames = np.where(lo.any(axis=1))[0]
    if len(bad_frames):
        lines.append(f"  涉及 {len(bad_frames)} 帧,最差帧 {int(np.argmin(conf.min(axis=1)))}")
    return "\n".join(lines)
