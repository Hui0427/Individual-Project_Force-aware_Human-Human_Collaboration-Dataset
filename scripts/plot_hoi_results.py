# scripts/plot_hoi_results.py
"""Plots for the hand-object distance results: per-hand distance curves with contact
segments, a fingertip contact timeline, and a 3D check at the closest-approach frame.

Usage: plot_hoi_results.py --out-dir outputs/hoi_drill_shot005 --mesh <obj> \
           --object drill --take shot_005/Drill.skel
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FINGERTIPS = ["ThumbEE", "IndexEE", "MiddleEE", "RingEE", "PinkyEE"]


def load(out_dir):
    data = defaultdict(dict)   # person -> joint -> (times, signed)
    per_frame_time = {}
    for r in csv.DictReader(open(out_dir / "distances.csv")):
        p, j, f = r["person"], r["joint"], int(r["frame"])
        data[p].setdefault(j, {})[f] = float(r["signed_dist_mm"])
        per_frame_time[f] = float(r["time"])
    times = np.array([per_frame_time[f] for f in sorted(per_frame_time)])
    out = {}
    for p, joints in data.items():
        out[p] = {j: np.array([v[f] for f in sorted(v)]) for j, v in joints.items()}
    return times, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--object", required=True)
    ap.add_argument("--take", required=True)
    ap.add_argument("--object-pose-name", default="drill_pose.csv",
                    help="object pose csv filename inside out-dir")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    summary = json.loads((out_dir / "summary.json").read_text())
    thr = summary["threshold_mm"]

    times, data = load(out_dir)

    # ---- Fig 1: min-over-joints signed distance per hand ----
    fig, axes = plt.subplots(len(data), 1, figsize=(13, 3.2 * len(data)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (person, joints) in zip(axes, sorted(data.items())):
        for side, color in (("Left", "tab:blue"), ("Right", "tab:red")):
            cols = [v for j, v in joints.items() if j.startswith(side)]
            if not cols:
                continue
            d = np.min(np.stack(cols), axis=0)
            n = min(len(times), len(d))
            ax.plot(times[:n], d[:n], color=color, lw=1.0, label=f"{side} hand (min joint)")
            m = d[:n] < thr
            ax.fill_between(times[:n], -60, 200, where=m, color=color, alpha=0.12)
        ax.axhline(thr, color="k", ls="--", lw=0.8, label=f"contact thr {thr:g} mm")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylim(-60, 200)
        ax.set_ylabel("signed dist (mm)")
        ax.set_title(f"{person} → {summary['object']}   (shaded = contact)")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_distance_curves.png", dpi=140)
    print("fig1_distance_curves.png")

    # ---- Fig 2: fingertip contact timeline ----
    rows, labels = [], []
    for person in sorted(data):
        for side in ("Left", "Right"):
            for tip in FINGERTIPS:
                j = f"{side}Hand{tip}"
                if j in data[person]:
                    rows.append(data[person][j] < thr)
                    labels.append(f"{person} {side[0]}-{tip[:-2]}")
    M = np.stack(rows)
    fig, ax = plt.subplots(figsize=(13, 0.32 * len(labels) + 1.5))
    n = min(M.shape[1], len(times))
    ax.imshow(M[:, :n], aspect="auto", cmap="Reds", interpolation="nearest",
              extent=[times[0], times[n - 1], len(labels) - 0.5, -0.5])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("time (s)")
    ax.set_title(f"fingertip contact (< {thr:g} mm)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_contact_timeline.png", dpi=140)
    print("fig2_contact_timeline.png")

    # ---- Fig 3: 3D check at the global closest-approach frame ----
    import trimesh
    from scipy.spatial.transform import Rotation
    offsets = json.loads((Path(__file__).resolve().parent.parent /
                          "outputs/mesh_offsets.json").read_text())[args.object]
    mf = offsets["takes"][args.take]["motive_frame"]
    mesh = trimesh.load(args.mesh, force="mesh")
    mesh.vertices = (mesh.vertices * np.array(offsets["scale"])) @ np.array(mf["R"]).T + np.array(mf["t_m"])
    surf = mesh.sample(3000)

    # frame of global minimum
    best_p, best_f, best_v = None, None, 1e9
    for person, joints in data.items():
        A = np.stack(list(joints.values()))
        f = int(np.unravel_index(A.argmin(), A.shape)[1])
        v = A.min()
        if v < best_v:
            best_p, best_f, best_v = person, f, v

    pose = list(csv.DictReader(open(out_dir / args.object_pose_name)))[best_f]
    t_o = np.array([float(pose["tx_mm"]), float(pose["ty_mm"]), float(pose["tz_mm"])]) / 1000
    R_o = Rotation.from_quat([float(pose["qx"]), float(pose["qy"]), float(pose["qz"]), float(pose["qw"])])

    jpos = {}
    for r in csv.DictReader(open(out_dir / f"{best_p}_hand_joints.csv")):
        if int(r["frame"]) == best_f:
            p_w = np.array([float(r["tx_mm"]), float(r["ty_mm"]), float(r["tz_mm"])]) / 1000
            jpos[r["joint"].split(":", 1)[-1]] = R_o.as_matrix().T @ (p_w - t_o)
    J = np.array(list(jpos.values()))

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*surf.T, s=1, c="lightgray", alpha=0.5)
    left = np.array([v for k, v in jpos.items() if k.startswith("Left")])
    right = np.array([v for k, v in jpos.items() if k.startswith("Right")])
    if len(left):
        ax.scatter(*left.T, s=25, c="tab:blue", label="left hand joints")
    if len(right):
        ax.scatter(*right.T, s=25, c="tab:red", label="right hand joints")
    allp = np.vstack([surf, J])
    c, r = allp.mean(0), (allp.max(0) - allp.min(0)).max() / 2
    for dim, cc in zip("xyz", c):
        getattr(ax, f"set_{dim}lim")(cc - r, cc + r)
    ax.set_title(f"{best_p} frame {best_f} (t={times[best_f]:.2f}s), "
                 f"min signed dist {best_v:.1f} mm — object-local frame")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_closest_frame_3d.png", dpi=140)
    print(f"fig3_closest_frame_3d.png  ({best_p} frame {best_f}, {best_v:.1f} mm)")


if __name__ == "__main__":
    main()
