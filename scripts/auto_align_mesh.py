# scripts/auto_align_mesh.py
"""Recover the mesh -> rigid-body alignment WITHOUT the manual Motive step.

Most objects in the August captures have no GeometryOffset/PitchYawRoll and the manual
alignment cannot currently be redone by hand, so we solve for it.

Marker-to-surface distance alone is underdetermined (3-5 constraints vs 6 DOF; free
6-DOF fits were measured to scatter by 12-27 mm). The gap is closed with the fact that
the object spends part of the take sitting still on a support surface:

  * Enumerate the mesh's physically STABLE resting poses (trimesh.poses.compute_stable_poses).
    Each one fixes the orientation up to a yaw about the vertical.
  * On the longest still interval, the rigid body's world rotation gives world-up
    expressed in the rigid-body frame; a candidate pose must map the mesh's resting
    normal onto it.
  * The mesh's lowest point must then sit on the support plane (1 translational DOF).

That leaves yaw + 2 horizontal translations against 3-5 marker constraints. Every
stable pose x yaw seed is optimised and ranked by marker residual, so the answer is
chosen by the data rather than assumed.

Validate before trusting: --object drill/chair/crate/... have a manual alignment, and
the residual here should land in the same 4-13 mm band that the manual ones do.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent


def load_pose(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    t = np.array([[float(r["tx_mm"]), float(r["ty_mm"]), float(r["tz_mm"])] for r in rows]) / 1000.0
    q = np.array([[float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])] for r in rows])
    return t, Rotation.from_quat(q)


def longest_still_run(t_w, R_w, speed_mm=2.0, ang_deg=0.5, min_run=15):
    v = np.r_[0, np.linalg.norm(np.diff(t_w, axis=0), axis=1)] * 1000
    w = np.r_[0, np.degrees(np.linalg.norm((R_w[:-1].inv() * R_w[1:]).as_rotvec(), axis=1))]
    still = (v < speed_mm) & (w < ang_deg)
    runs, start = [], None
    for i, s in enumerate(np.r_[still, False]):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if i - start >= min_run:
                runs.append((start, i))
            start = None
    if not runs:
        return None, []
    return max(runs, key=lambda r: r[1] - r[0]), runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True, help="key in motive_props.json")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--object-pose", required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="mesh scale factor")
    ap.add_argument("--symmetric", choices=["none", "yaw"], default="none")
    ap.add_argument("--max-poses", type=int, default=6, help="stable poses to try")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    props = json.loads((ROOT / "outputs/motive_props.json").read_text())["objects"][args.object]
    M = np.array([m["offset_m"] for m in props["markers"]])
    rad = np.array([m["diameter_m"] / 2 for m in props["markers"]])

    mesh = trimesh.load(args.mesh, force="mesh")
    if args.scale != 1.0:
        mesh.vertices = mesh.vertices * args.scale
    print(f"{args.object}: {len(M)} markers | mesh extents {np.round(mesh.extents, 3)} m")

    t_w, R_w = load_pose(args.object_pose)
    run, runs = longest_still_run(t_w, R_w)
    if run is None:
        raise SystemExit("no still interval found; cannot use the support constraints")
    idx = np.arange(*run)
    ups = np.stack([R.T @ np.array([0, 1.0, 0]) for R in R_w[idx].as_matrix()])
    up_rb = ups.mean(0); up_rb /= np.linalg.norm(up_rb)
    spread = np.degrees(np.arccos(np.clip(ups @ up_rb, -1, 1))).std()
    print(f"still runs: {len(runs)}, using the longest ({run[1]-run[0]} frames, "
          f"{run[0]}-{run[1]}); up-in-rb {np.round(up_rb, 3)} (spread {spread:.2f} deg)")

    poses, probs = trimesh.poses.compute_stable_poses(mesh)
    poses, probs = poses[:args.max_poses], probs[:args.max_poses]
    print(f"stable resting poses: {len(poses)} (probabilities {np.round(probs, 3).tolist()})")

    samp, face = trimesh.sample.sample_surface(mesh, 200_000, seed=0)
    tree = cKDTree(samp)
    normals = mesh.face_normals[face]
    sub = samp[::40]
    pivot_h = float(t_w[idx][:, 1].mean())

    def residuals(R, t):
        p = (M - t) @ R
        d, i = tree.query(p)
        sgn = np.sign(np.einsum("ij,ij->i", p - samp[i], normals[i]))
        return (sgn * d - rad) * 1000

    results = []
    yaw_seeds = [0.0] if args.symmetric == "yaw" else np.linspace(0, 2 * np.pi, 8, endpoint=False)
    for pi, (P, prob) in enumerate(zip(poses, probs)):
        # P maps the mesh so that its resting face is down and +Z is up. Compose with the
        # rotation that carries +Z onto the measured world-up direction in the rb frame.
        R_rest = P[:3, :3]
        R_z2up = Rotation.align_vectors(up_rb[None], np.array([[0, 0, 1.0]]))[0].as_matrix()
        R_base = R_z2up @ R_rest
        # support height: the mesh bottom must sit on the plane the object rests on
        low_base = ((samp @ R_base.T) @ up_rb).min()

        def build(x):
            psi, tx, ty, tz = x
            R = Rotation.from_rotvec(up_rb * psi).as_matrix() @ R_base
            return R, np.array([tx, ty, tz])

        def cost(x):
            R, t = build(x)
            r = residuals(R, t)
            low = ((sub @ R.T + t) @ up_rb).min()
            # the pivot is the marker centroid, so the support height is implied by the
            # geometry itself: penalise the mesh bottom drifting off the plane it rested on
            return float((r ** 2).sum() + (low - low_base) ** 2 * 1e6)

        best = None
        for psi0 in yaw_seeds:
            r = minimize(cost, np.array([psi0, *(-M.mean(0))]), method="Nelder-Mead",
                         options={"xatol": 1e-5, "fatol": 1e-9, "maxiter": 4000})
            if best is None or r.fun < best.fun:
                best = r
        R, t = build(best.x)
        res = residuals(R, t)
        results.append((np.abs(res).mean(), pi, float(prob), R, t, res))

    results.sort(key=lambda r: r[0])
    print(f"\n{'rank':>4} {'pose':>4} {'prob':>6} {'mean|resid|':>12}  per-marker residual (mm)")
    for k, (score, pi, prob, R, t, res) in enumerate(results[:4]):
        print(f"{k:>4} {pi:>4} {prob:>6.3f} {score:>10.1f}mm   {np.round(res, 1).tolist()}")

    score, pi, prob, R, t, res = results[0]
    pyr = Rotation.from_matrix(R).as_euler("XYZ", degrees=True)
    margin = results[1][0] / score if len(results) > 1 else float("inf")
    print(f"\nbest: stable pose {pi} (prob {prob:.3f})")
    print(f"  pitch/yaw/roll = {np.round(pyr, 2).tolist()} deg")
    print(f"  offset         = {np.round(t, 4).tolist()} m")
    print(f"  mean |residual| = {score:.1f} mm   (manual alignments sit at 4-13 mm)")
    print(f"  margin over runner-up: {margin:.1f}x "
          f"{'(decisive)' if margin > 2 else '(AMBIGUOUS - check by eye)'}")
    if args.symmetric == "yaw":
        print("  yaw is arbitrary (object declared rotationally symmetric)")

    out = {
        "object": args.object, "mesh": args.mesh,
        "method": "auto: stable-pose enumeration + marker surface fit + support plane",
        "pitch_yaw_roll_deg": pyr.tolist(), "offset_m": t.tolist(),
        "scale_factor": [args.scale] * 3,
        "marker_residual_mm": res.tolist(), "mean_abs_residual_mm": float(score),
        "stable_pose_index": int(pi), "stable_pose_prob": prob,
        "margin_over_runner_up": float(margin),
        "symmetric": args.symmetric,
        "still_run": [int(run[0]), int(run[1])],
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
