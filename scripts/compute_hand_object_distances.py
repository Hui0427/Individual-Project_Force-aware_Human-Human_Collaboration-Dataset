# scripts/compute_hand_object_distances.py
"""Per-frame hand-joint -> object-surface distances, contact labels, penetration.

Method (with sources):
  - Distance: point-to-mesh nearest distance, computed in the OBJECT's Hips-local frame
    so the mesh and its spatial index are built once. Hand joints are transformed into
    that frame per frame: p_local = R_hips^T (p_world - t_hips).
  - Contact: proximity threshold, following GRAB (Taheri et al., ECCV 2020), which uses
    4.5 mm for SKIN-SURFACE vertices. Our probes are JOINT CENTRES, so the threshold
    must additionally absorb finger radius (~8-10 mm) and the calibration floor
    measured for this pipeline (~10-30 mm depending on object). Default 15 mm, with a
    sensitivity sweep reported at 10/15/20 mm.
  - Signed distance: sign from dot(p - nearest_point, face_normal). The YCB meshes are
    not watertight, so ray/winding inside tests are unreliable; the normal heuristic is
    exact near the surface, which is the only regime where the sign matters.
  - No pose is modified anywhere (annotation only, per project plan).

Everything runs in the Captury right-handed frame in metres; Unity/handedness never
enters distance computation.

Usage:
  compute_hand_object_distances.py --object drill --take shot_005/Drill.skel \
      --mesh <obj path> --object-pose <csv> --hands <csv> [<csv> ...] --out-dir <dir>
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

OFFSETS_PATH = Path(__file__).resolve().parent.parent / "outputs/mesh_offsets.json"

FINGERTIPS = ["ThumbEE", "IndexEE", "MiddleEE", "RingEE", "PinkyEE"]


def load_object_pose(path):
    rows = list(csv.DictReader(open(path)))
    t = np.array([[float(r["tx_mm"]), float(r["ty_mm"]), float(r["tz_mm"])] for r in rows]) / 1000.0
    q = np.array([[float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])] for r in rows])
    times = np.array([float(r["time"]) for r in rows])
    return t, Rotation.from_quat(q), times


def load_hand_joints(path):
    """Return (joint_names, positions[frame, joint, 3] in metres)."""
    frames = {}
    names = []
    for r in csv.DictReader(open(path)):
        f = int(r["frame"])
        j = r["joint"].split(":", 1)[-1]
        if f == 0:
            names.append(j)
        frames.setdefault(f, {})[j] = (
            float(r["tx_mm"]) / 1000.0, float(r["ty_mm"]) / 1000.0, float(r["tz_mm"]) / 1000.0)
    n_frames = len(frames)
    P = np.array([[frames[f][j] for j in names] for f in range(n_frames)])
    return names, P


def contact_segments(mask, times):
    """Consecutive-True runs -> [(start_frame, end_frame, t0, t1)]."""
    segs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            segs.append((start, i - 1, float(times[start]), float(times[i - 1])))
            start = None
    if start is not None:
        segs.append((start, len(mask) - 1, float(times[start]), float(times[-1])))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True)
    ap.add_argument("--take", required=True, help="key in mesh_offsets.json takes{}")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--object-pose", required=True)
    ap.add_argument("--hands", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threshold-mm", type=float, default=15.0)
    ap.add_argument("--sweep-mm", nargs="*", type=float, default=[10.0, 15.0, 20.0])
    ap.add_argument("--penetration-mm", type=float, default=5.0,
                    help="flag frames with signed distance below -this")
    ap.add_argument("--jump-mm", type=float, default=30.0,
                    help="flag per-joint frame-to-frame distance jumps above this")
    ap.add_argument("--sampled", type=int, default=-1, metavar="N",
                    help="Query a KD-tree of N surface samples instead of exact "
                         "point-to-triangle. Overestimates distance by at most the "
                         "sample spacing (measured 0.02+-0.03 mm at N=500k, max 0.22 mm), "
                         "negligible against a 15 mm contact threshold. "
                         "0 forces exact; -1 (default) picks automatically: exact for "
                         "scanned YCB meshes (~16k faces), sampled for the Tripo meshes "
                         "(~2M faces), where exact is intractable and the denser surface "
                         "makes sampling more accurate anyway.")
    ap.add_argument("--auto-sample-threshold", type=int, default=200_000,
                    help="face count above which --sampled -1 switches to sampling")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- mesh -> object Hips-local frame (calibration result, held fixed) ---
    offsets = json.loads(OFFSETS_PATH.read_text())[args.object]
    S = np.array(offsets["scale"])
    mf = offsets["takes"][args.take]["motive_frame"]
    R_geo, t_geo = np.array(mf["R"]), np.array(mf["t_m"])

    mesh = trimesh.load(args.mesh, force="mesh")
    mesh.vertices = (mesh.vertices * S) @ R_geo.T + t_geo

    n_sampled = args.sampled
    if n_sampled < 0:
        n_sampled = 500_000 if len(mesh.faces) > args.auto_sample_threshold else 0
        print(f"mesh has {len(mesh.faces):,} faces -> "
              f"{'sampled' if n_sampled else 'exact'} mode (auto)")
    args.sampled = n_sampled

    if args.sampled > 0:
        from scipy.spatial import cKDTree
        samp_pts, samp_face = trimesh.sample.sample_surface(mesh, args.sampled, seed=0)
        tree = cKDTree(samp_pts)

        def query(points):
            d, i = tree.query(points)
            return samp_pts[i], d, samp_face[i]
        mode = f"sampled KD-tree ({args.sampled} pts)"
    else:
        pq = trimesh.proximity.ProximityQuery(mesh)

        def query(points):
            return pq.on_surface(points)
        mode = "exact point-to-triangle"

    print(f"mesh: {len(mesh.faces)} faces, aligned into {args.object} Hips frame "
          f"(take {args.take}, scale {S.tolist()}), distance mode: {mode}")

    t_obj, R_obj, times = load_object_pose(args.object_pose)

    all_rows = []
    summary = {"object": args.object, "take": args.take,
               "threshold_mm": args.threshold_mm, "sweep_mm": args.sweep_mm,
               "penetration_mm": args.penetration_mm, "persons": {}}

    for hand_path in args.hands:
        person = Path(hand_path).stem.split("_")[0]
        names, P_world = load_hand_joints(hand_path)
        n = min(len(P_world), len(t_obj))
        P_world, Rn, tn, tms = P_world[:n], R_obj[:n], t_obj[:n], times[:n]

        # world -> object-local, all frames at once
        Rm = Rn.as_matrix()                        # (n,3,3)
        P_loc = np.einsum("nij,nkj->nki", Rm.transpose(0, 2, 1), P_world - tn[:, None, :])

        flat = P_loc.reshape(-1, 3)
        closest, dist, tri = query(flat)
        normals = mesh.face_normals[tri]
        sign = np.sign(np.einsum("ij,ij->i", flat - closest, normals))
        sign[sign == 0] = 1.0
        signed = (sign * dist).reshape(n, len(names)) * 1000.0   # mm
        closest = closest.reshape(n, len(names), 3)

        thr = args.threshold_mm
        contact = signed < thr
        pen = signed < -args.penetration_mm
        jumps = np.abs(np.diff(signed, axis=0))
        anomaly_frames = sorted(set(np.where(jumps > args.jump_mm)[0].tolist()))

        pers = {"n_frames": n, "joints": {}, "sweep_contact_frames": {},
                "penetration_frames": sorted(set(np.where(pen.any(1))[0].tolist())),
                "anomaly_frames": anomaly_frames}
        for k in args.sweep_mm:
            pers["sweep_contact_frames"][f"{k:g}mm"] = int((signed < k).any(1).sum())

        for ji, jn in enumerate(names):
            segs = contact_segments(contact[:, ji], tms)
            pers["joints"][jn] = {
                "min_signed_mm": float(signed[:, ji].min()),
                "contact_frames": int(contact[:, ji].sum()),
                "contact_segments": segs,
            }
        summary["persons"][person] = pers

        for f in range(n):
            for ji, jn in enumerate(names):
                all_rows.append([f, round(float(tms[f]), 6), person, jn,
                                 round(float(signed[f, ji]), 2), int(contact[f, ji]),
                                 *np.round(closest[f, ji], 5)])
        print(f"{person}: {n} frames x {len(names)} joints | "
              f"min signed {signed.min():.1f} mm | "
              f"contact frames @{thr:g}mm: {int(contact.any(1).sum())} | "
              f"penetration(<-{args.penetration_mm:g}mm) frames: {len(pers['penetration_frames'])}")

    with open(out_dir / "distances.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time", "person", "joint", "signed_dist_mm", "contact",
                    "closest_x_local", "closest_y_local", "closest_z_local"])
        w.writerows(all_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}/distances.csv and summary.json")


if __name__ == "__main__":
    main()
