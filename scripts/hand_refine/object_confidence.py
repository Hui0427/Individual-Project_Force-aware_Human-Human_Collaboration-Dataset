# scripts/hand_refine/object_confidence.py
"""Per-frame confidence in the tracked OBJECT pose, and a proximity gate.

The object is not tracked reliably at all times - it can jump or drift when markers are
occluded. Without a gate the penetration term happily shoves a perfectly good hand out
of the way of a mistracked object, i.e. bad object data corrupts good hand data.

Two independent gates, multiplied:

  object confidence  robust jump/jerk test on the object's own trajectory
                     (translation and rotation), so teleports switch the physics off
  proximity gate     the physical terms only act when the hand is actually near the
                     object; far away, the object should have no say at all. Computed
                     from the ORIGINAL pose so it is fixed input data - a gate that
                     moved with the optimiser could be escaped by pushing the hand away.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def _mad(x):
    return np.median(np.abs(x - np.median(x))) * 1.4826 + 1e-9


def object_confidence(t_m, R, k=6.0, floor_mm=25.0, floor_deg=10.0):
    """t_m: (F,3) metres, R: Rotation of length F. Returns (F,) confidence in [0,1].

    A frame is doubted when the object's motion is both statistically unusual for this
    take and physically implausible in absolute terms (floor_mm per frame ~ 1.5 m/s at
    60 fps for a carried object).
    """
    F = len(t_m)
    v = np.r_[0, np.linalg.norm(np.diff(t_m, axis=0), axis=1)] * 1000
    w = np.r_[0, np.degrees(np.linalg.norm((R[:-1].inv() * R[1:]).as_rotvec(), axis=1))]
    av = np.abs(np.r_[0, np.diff(v)])

    exc = np.zeros(F)
    for sig, floor in ((v, floor_mm), (w, floor_deg), (av, floor_mm)):
        scale = max(_mad(sig) * k, floor)
        exc += np.clip((sig - np.median(sig)) / scale - 1.0, 0, None)
    return 1.0 / (1.0 + exc)


def proximity_gate(min_dist_mm, near_mm=20.0, far_mm=80.0):
    """1 while the hand is within near_mm of the surface, fading to 0 at far_mm.

    Linear ramp rather than a hard cut so the loss stays continuous; beyond far_mm the
    object exerts no force on the hand at all.
    """
    g = (far_mm - min_dist_mm) / (far_mm - near_mm)
    return np.clip(g, 0.0, 1.0)


def detect_duplicate_frames(t_m, tol_mm=1e-3):
    """Fraction of frames that merely repeat their predecessor.

    The Captury exports here are 30 fps but sampled at 60, so every real sample appears
    twice and the object advances in a staircase: hold, jump, hold. Distances stay
    correct but the jumps make the physical terms chatter, and the hands - which the
    tracker interpolates - do not have the same staircase, so the two sources disagree
    in time.
    """
    if len(t_m) < 2:
        return 0.0
    dup = np.abs(np.diff(t_m, axis=0)).max(axis=1) < tol_mm / 1000.0
    return float(dup.mean())


def smooth_object_pose(t_m, R, window=5):
    """Resample a staircased object trajectory onto a smooth one.

    Translation: centred moving average. Rotation: SLERP through the same window via
    rotation-vector averaging relative to each frame, which is stable for the small
    inter-frame angles here (<7 deg/frame measured).

    Applied to the pose the physical terms see; the raw pose stays untouched in the
    exported annotations so nothing downstream is silently altered.
    """
    from scipy.ndimage import uniform_filter1d
    t_s = uniform_filter1d(t_m, size=window, axis=0, mode="nearest")

    F = len(R)
    half = window // 2
    q_s = np.zeros((F, 4))
    Rm = R.as_matrix()
    for i in range(F):
        lo, hi = max(0, i - half), min(F, i + half + 1)
        # average in the tangent space at frame i, then map back
        rel = (R[i].inv() * R[lo:hi]).as_rotvec().mean(axis=0)
        q_s[i] = (R[i] * Rotation.from_rotvec(rel)).as_quat()
    return t_s, Rotation.from_quat(q_s)


def summarize(obj_conf, gate, thr=0.5):
    active = obj_conf * gate
    return (f"object confidence: {(obj_conf < thr).sum()} frames doubted "
            f"(mean {obj_conf.mean():.3f}) | "
            f"proximity gate open on {(gate > 0).sum()} frames "
            f"(fully open {(gate >= 1).sum()}) | "
            f"physics active on {(active > 0.1).sum()}/{len(active)} frames")
