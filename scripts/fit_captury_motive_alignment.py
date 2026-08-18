# scripts/fit_captury_motive_alignment.py
"""Solve the transform between Motive's rigid-body frame and Captury's Hips frame.

Captury's per-object .skel files list the Motive markers with their rest-pose positions
in millimetres, in the same frame as the object's Root (= the `<obj>:Hips` node the FBX
animates). The .motive profile lists the same markers as offsets from the rigid-body
pivot in metres. Same physical points, two frames -> the transform is recoverable
exactly by Kabsch, no documentation guesswork needed.

The "motive_N" names in the .skel are NOT unique (Drill has three `motive_1`, Table five
`motive_0`), so they carry no correspondence. We brute-force the assignment instead and
keep the one with the lowest fit residual; with 3-6 well-separated markers the correct
correspondence wins by a wide margin.

Crucially this also settles handedness: if the reflection-allowed fit beats the proper
rotation, the two frames differ by a mirror - the systematic error suspected in Unity.
"""
import itertools
import json
import re
from pathlib import Path

import numpy as np

PROPS_PATH = Path("outputs/motive_props.json")
OUT_PATH = Path("outputs/captury_motive_alignment.json")

DESKTOP = Path("/Users/hxy/Desktop")

# One entry per (object, take). The rigid-body definition drifts between recording
# sessions - always as a pure yaw, since the body gets rebuilt with the object sitting
# at a different heading - so every take needs its own fit. Within a session the fits
# agree to 0.01 deg, so takes from one session are interchangeable.
#
# shot_001/002/003 export objects under generic names; the mapping below comes from
# identify_unnamed_props.py, which fingerprints them by pairwise marker distances
# (prop=jug, prop-2=drill, prop-3=apple).
SKEL_FILES = {
    "jug": [
        DESKTOP / "YCB/shot_001/prop.skel",
        DESKTOP / "YCB/shot_002/prop.skel",
        DESKTOP / "YCB/shot_003/prop.skel",
        DESKTOP / "core4d_refine_project/data/shot_005/jug.skel",
    ],
    "drill": [
        DESKTOP / "YCB/shot_001/prop-2.skel",
        DESKTOP / "YCB/shot_002/prop-2.skel",
        DESKTOP / "YCB/shot_003/prop-2.skel",
        DESKTOP / "YCB/align_drill_YCB/Assets/shot_005/Drill.skel",
    ],
    "table": [
        DESKTOP / "YCB/align_drill_YCB/Assets/shot_005/Table.skel",
        DESKTOP / "YCB/5stardata/shot_018/table.skel",
        DESKTOP / "YCB/5stardata/shot_031/table.skel",
        DESKTOP / "YCB/0728_data/0728_shot_007/Table.skel",
        DESKTOP / "YCB/0728_data/0728_shot_020/Table.skel",
    ],
    "crate": [
        DESKTOP / "YCB/new_data_0716/shot_012/Crate.skel",
        DESKTOP / "YCB/0728_data/0728_shot_020/Crate.skel",
    ],
    "chair": [DESKTOP / "YCB/5stardata/shot_031/chair.skel"],
    "football": [DESKTOP / "YCB/5stardata/shot_018/football.skel"],
    "hammer": [DESKTOP / "YCB/0728_data/0728_shot_007/Hammer.skel"],
    "spray": [DESKTOP / "YCB/0728_data/0728_shot_020/Spray.skel"],
}


def parse_skel(path):
    """Return (root_xyz_mm, [xyz_mm, ...]) from a Captury .skel, markers in file order."""
    root = None
    markers = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "Root_tx" and root is None:
            root = np.array([float(v) for v in parts[3:6]])
        if re.match(r"motive_\d+$", parts[0]):
            markers.append(np.array([float(v) for v in parts[2:5]]))
    return root, markers


def kabsch(P, Q, allow_reflection=False):
    """Least-squares R,t with Q ~ R@P + t. Returns (R, t, rms, det)."""
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.eye(3)
    if not allow_reflection:
        D[2, 2] = d
    R = Vt.T @ D @ U.T
    t = Qc - R @ Pc
    rms = float(np.sqrt(((Q - (P @ R.T + t)) ** 2).sum(1).mean()))
    return R, t, rms, float(np.linalg.det(R))


def main():
    props = json.loads(PROPS_PATH.read_text())["objects"]
    out = {}

    for name, skel_paths in SKEL_FILES.items():
        out[name] = {}
        print(f"\n=== {name} ===")
        print(f"  {'take':34s} {'mk':>3s} {'rot':>9s} {'axis':>18s} {'rms':>7s}  handedness")

        for skel_path in skel_paths:
            if not skel_path.exists():
                print(f"  [skip] no skel at {skel_path}")
                continue
            root_mm, markers_mm = parse_skel(skel_path)
            motive_markers = props[name]["markers"]

            # Captury marker positions are global rest-pose; make them Hips-local, in m.
            Q_all = np.array([(m - root_mm) / 1000.0 for m in markers_mm])
            all_motive = np.array([m["offset_m"] for m in motive_markers])

            # The two sides can disagree on marker count in either direction: a take may
            # drop an occluded marker (chair: 4 vs 5), or the rigid body may have gained
            # one since the profile was saved (table: 6 vs 5). Match on the largest
            # common subset, searching subsets of whichever side is bigger.
            k = min(len(Q_all), len(all_motive))
            best = best_refl = None
            for q_idx in itertools.combinations(range(len(Q_all)), k):
                Q_sub = Q_all[list(q_idx)]
                for perm in itertools.permutations(range(len(all_motive)), k):
                    P = all_motive[list(perm)]
                    cand = kabsch(P, Q_sub) + (perm, q_idx)
                    cand_r = kabsch(P, Q_sub, allow_reflection=True) + (perm, q_idx)
                    if best is None or cand[2] < best[2]:
                        best = cand
                    if best_refl is None or cand_r[2] < best_refl[2]:
                        best_refl = cand_r

            R, t, rms, det, perm, q_idx = best
            _, _, rms_r, det_r, _, _ = best_refl
            Q = Q_all[list(q_idx)]

            # Report the rotation as an axis-angle, which is far easier to sanity-check
            # than a matrix when the answer is expected to be a plain yaw.
            ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
            w, v = np.linalg.eig(R)
            axis = np.real(v[:, np.argmin(np.abs(w - 1))])
            axis = axis / np.linalg.norm(axis)

            # 3 points are always coplanar, so a reflection fits them just as well as a
            # rotation; the handedness test only means anything from 4 markers up.
            verdict = "degenerate (3 coplanar markers)" if len(Q) < 4 else (
                "MIRRORED" if rms_r < rms - 1e-9 else "same handedness")

            take = f"{skel_path.parent.name}/{skel_path.name}"
            out[name][take] = {
                "skel": str(skel_path),
                "n_markers": len(Q),
                "correspondence_captury_to_motive": list(perm),
                "captury_markers_used": list(q_idx),
                "R_motive_to_captury": R.round(6).tolist(),
                "t_motive_to_captury_m": t.round(6).tolist(),
                "rms_mm": rms * 1000,
                "det": det,
                "angle_deg": float(ang),
                "axis": axis.round(4).tolist(),
                "rms_mm_if_reflection_allowed": rms_r * 1000,
                "handedness_verdict": verdict,
            }

            print(f"  {take[:34]:34s} {len(Q):>3d} {ang:>8.2f}° "
                  f"{str(axis.round(2)):>18s} {rms*1000:>6.2f}mm  {verdict}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
