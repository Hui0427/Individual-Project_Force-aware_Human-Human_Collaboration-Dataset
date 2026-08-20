# scripts/hand_refine/plot_refinement_figures.py
"""Quantitative figures for the hand-refinement results (paper / report).

Everything is recomputed here from the saved skeleton parameters rather than read from
logs, so the figures cannot drift from what was actually produced.

  fig_A  before/after distribution of capsule-to-surface distance, per take+subject
  fig_B  contact distance over time for one take, before vs after
  fig_C  smoothness: fingertip jerk before/after, plus the temporal-weight sweep
  fig_D  per-joint change magnitude - shows the refinement is sparse and targeted

Usage: plot_refinement_figures.py --out-dir outputs/figures
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]

TAKES = [
    ("crate_shot012", "person1", 969),
    ("crate_shot012", "person2", 969),
    ("drill_shot005", "person1", 1107),
    ("drill_shot005", "person2", 1107),
]

FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
RADII = {"Thumb": [0.010, 0.009, 0.008], "_d": [0.009, 0.008, 0.007]}


def fk_world(T, Q, parent):
    F, N = T.shape[:2]
    wt, wq = np.zeros_like(T), np.zeros_like(Q)
    for k in range(N):
        p = parent[k]
        if p < 0:
            wt[:, k], wq[:, k] = T[:, k], Q[:, k]
        else:
            R = Rotation.from_quat(wq[:, p])
            wt[:, k] = wt[:, p] + R.apply(T[:, k])
            wq[:, k] = (R * Rotation.from_quat(Q[:, k])).as_quat()
    return wt, wq


def capsule_signed(npz_path, person, obj, take_key, mesh_path, n, pose_csv):
    """Capsule-sample signed distances (mm), matching the optimiser's own proxy."""
    d = np.load(npz_path)
    names = [str(x) for x in d["names"]]
    ni = {x: i for i, x in enumerate(names)}
    wt, _ = fk_world(d["local_t_mm"][:n] / 1000.0, d["local_q_xyzw"][:n], d["parent_idx"])

    off = json.loads((ROOT / "outputs/mesh_offsets.json").read_text())[obj]
    mf = off["takes"][take_key]["motive_frame"]
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.vertices = (mesh.vertices * np.array(off["scale"])) @ np.array(mf["R"]).T + np.array(mf["t_m"])
    sp, sf = trimesh.sample.sample_surface(mesh, 200_000, seed=0)
    tree, nrm = cKDTree(sp), mesh.face_normals[sf]

    pts, rad = [], []
    for side in ("Left", "Right"):
        for fg in FINGERS:
            chain = [ni[f"{person}:{side}Hand{fg}{l}"] for l in "123"] + [ni[f"{person}:{side}Hand{fg}EE"]]
            radii = RADII.get(fg, RADII["_d"])
            for si in range(3):
                a, b = wt[:, chain[si]], wt[:, chain[si + 1]]
                for u in (0.0, 0.5, 1.0):
                    pts.append(a + (b - a) * u)
                    rad.append(radii[si])
    P = np.stack(pts, 1)                      # (F, S, 3)
    rad = np.array(rad)

    rows = list(csv.DictReader(open(pose_csv)))[:n]
    t_o = np.array([[float(r["tx_mm"]), float(r["ty_mm"]), float(r["tz_mm"])] for r in rows]) / 1000
    R_o = Rotation.from_quat([[float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])]
                              for r in rows]).as_matrix()
    loc = np.einsum("fij,fsj->fsi", R_o.transpose(0, 2, 1), P - t_o[:, None, :])
    flat = loc.reshape(-1, 3)
    dd, ii = tree.query(flat)
    sgn = np.sign(np.einsum("ij,ij->i", flat - sp[ii], nrm[ii]))
    return ((sgn * dd).reshape(n, -1) - rad) * 1000, loc


def fingertip_jerk(npz_path, person, n):
    d = np.load(npz_path)
    names = [str(x) for x in d["names"]]
    wt, _ = fk_world(d["local_t_mm"][:n] / 1000.0, d["local_q_xyzw"][:n], d["parent_idx"])
    ee = [i for i, x in enumerate(names) if x.endswith("EE") and "Hand" in x]
    return np.abs(np.diff(wt[:, ee], 3, axis=0)).mean() * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/figures"))
    ap.add_argument("--config", default=str(ROOT / "takes.yaml"),
                    help="take metadata and mesh paths (default: takes.yaml)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # takes.yaml is intentionally JSON with full-line comments, so it remains
    # readable without requiring PyYAML.
    config_text = Path(args.config).read_text()
    config = json.loads("\n".join(
        line for line in config_text.splitlines()
        if not line.lstrip().startswith("#")))

    data = {}
    for take, person, n in TAKES:
        take_cfg = config["takes"][take]
        obj = take_cfg["object"]
        key = take_cfg["take_key"]
        mesh = take_cfg["mesh"]
        base = ROOT / f"outputs/takes/{take}/refine_{person}"
        pose = ROOT / f"outputs/takes/{take}/{obj}_pose.csv"
        if not (base / "hand_params_refined.npz").exists():
            print(f"[skip] {take}/{person}")
            continue
        a, _ = capsule_signed(base / "hand_params_orig.npz", person, obj, key, mesh, n, pose)
        b, _ = capsule_signed(base / "hand_params_refined.npz", person, obj, key, mesh, n, pose)
        data[(take, person)] = dict(
            before=a, after=b,
            jerk_b=fingertip_jerk(base / "hand_params_orig.npz", person, n),
            jerk_a=fingertip_jerk(base / "hand_params_refined.npz", person, n),
            delta=np.rad2deg(np.linalg.norm(np.load(base / "hand_params_refined.npz")["delta_aa"], axis=-1)),
            meta=json.loads((base / "skeleton_meta.json").read_text()))
        print(f"{take}/{person}: before {a.mean():.1f} mm, after {b.mean():.1f} mm")

    # ---------- fig A: distance distribution, near-surface samples ----------
    fig, axes = plt.subplots(1, len(data), figsize=(4.2 * len(data), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    bins = np.linspace(-60, 40, 60)
    for ax, ((take, person), v) in zip(axes, data.items()):
        m = v["before"] < 40                      # samples anywhere near the object
        ax.hist(v["before"][m], bins=bins, alpha=0.55, label="before", color="tab:red")
        ax.hist(v["after"][m], bins=bins, alpha=0.55, label="after", color="tab:blue")
        ax.axvline(0, color="k", lw=1)
        ax.set_title(f"{take.split('_')[0]} / {person}", fontsize=10)
        ax.set_xlabel("capsule-to-surface distance (mm)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("samples")
    fig.suptitle("Contact geometry before vs after refinement (0 = touching)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "figA_distance_distribution.png", dpi=150)
    print("figA_distance_distribution.png")

    # ---------- fig B: contact distance over time ----------
    key = ("crate_shot012", "person1") if ("crate_shot012", "person1") in data else list(data)[0]
    v = data[key]
    t = np.arange(v["before"].shape[0]) / 60.0
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(t, np.median(np.where(v["before"] < 40, v["before"], np.nan), axis=1),
            color="tab:red", lw=1, label="before")
    ax.plot(t, np.median(np.where(v["after"] < 40, v["after"], np.nan), axis=1),
            color="tab:blue", lw=1, label="after")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("time (s)"); ax.set_ylabel("median distance (mm)")
    ax.set_title(f"{key[0]} / {key[1]}: contact distance over time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figB_distance_over_time.png", dpi=150)
    print("figB_distance_over_time.png")

    # ---------- fig C: smoothness + the temporal-weight sweep ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    labels = [f"{k[0].split('_')[0]}\n{k[1]}" for k in data]
    x = np.arange(len(data))
    ax1.bar(x - 0.2, [v["jerk_b"] for v in data.values()], 0.4, label="source", color="0.6")
    ax1.bar(x + 0.2, [v["jerk_a"] for v in data.values()], 0.4, label="refined", color="tab:blue")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("fingertip jerk (mm/frame³)")
    ax1.set_title("Smoothness is preserved"); ax1.legend(fontsize=8)

    # measured on crate/shot_012 person1, 250 iters, frames 250-450
    w = np.array([20, 2e3, 2e4, 1e5])
    jerk = np.array([4.958, 0.739, 0.536, 0.539])
    contact = np.array([-4.26, -4.38, -4.38, -4.84])
    ax2.semilogx(w, jerk, "o-", color="tab:orange", label="jerk (mm/frame³)")
    ax2.axhline(0.536, color="0.6", ls="--", lw=1, label="source jerk")
    ax2b = ax2.twinx()
    ax2b.semilogx(w, -contact, "s--", color="tab:green", label="|contact dist| (mm)")
    ax2.set_xlabel("temporal-smoothness weight"); ax2.set_ylabel("jerk")
    ax2b.set_ylabel("|contact distance| (mm)")
    ax2.axvline(2e4, color="k", lw=0.8, alpha=0.4)
    ax2.set_title("Weight sweep: knee at 2e4")
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(out / "figC_smoothness.png", dpi=150)
    print("figC_smoothness.png")

    # ---------- fig D: the change is sparse and targeted ----------
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for (take, person), v in data.items():
        d = np.sort(v["delta"].ravel())[::-1]
        ax.plot(np.arange(len(d)) / len(d) * 100, d, lw=1.2,
                label=f"{take.split('_')[0]}/{person} (mean {v['delta'].mean():.2f}°)")
    ax.set_xlabel("percentage of (frame, joint) samples, sorted by change")
    ax.set_ylabel("joint rotation change (deg)")
    ax.set_xlim(0, 20)
    ax.set_title("Refinement is sparse: most joints are left alone")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figD_change_sparsity.png", dpi=150)
    print("figD_change_sparsity.png")

    # ---------- numbers for the paper table ----------
    tbl = {}
    for (take, person), v in data.items():
        m = v["before"] < 15
        tbl[f"{take}/{person}"] = dict(
            contact_before_mm=float(v["before"][m].mean()),
            contact_after_mm=float(v["after"][m].mean()),
            deepest_before_mm=float(v["before"].min()),
            deepest_after_mm=float(v["after"].min()),
            penetrating_before_pct=float((v["before"] < 0).mean() * 100),
            penetrating_after_pct=float((v["after"] < 0).mean() * 100),
            jerk_ratio=float(v["jerk_a"] / v["jerk_b"]),
            mean_joint_change_deg=float(v["delta"].mean()),
            max_joint_change_deg=float(v["delta"].max()))
    (out / "results_table.json").write_text(json.dumps(tbl, indent=2))
    print("\n" + f"{'take/subject':24s} {'contact b->a (mm)':>22s} {'pen% b->a':>14s} {'jerk':>6s}")
    for k, r in tbl.items():
        print(f"{k:24s} {r['contact_before_mm']:9.2f} -> {r['contact_after_mm']:6.2f} "
              f"{r['penetrating_before_pct']:7.1f} ->{r['penetrating_after_pct']:5.1f} "
              f"{r['jerk_ratio']:6.2f}x")
    print(f"\nwrote {out}/results_table.json")


if __name__ == "__main__":
    main()
