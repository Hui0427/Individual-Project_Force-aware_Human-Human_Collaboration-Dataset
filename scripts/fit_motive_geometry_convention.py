# scripts/fit_motive_geometry_convention.py
"""Determine empirically how Motive composes GeometryScale / PitchYawRoll / Offset.

Motive's docs do not state the Euler order or the order of operations for the manual
mesh-alignment fields, so we recover it from data instead of guessing.

Physical constraint used as ground truth: the mocap markers are glued to the real
object's surface. So for the *correct* convention, every marker offset (given in the
rigid-body frame) must land ~one marker-radius off the aligned mesh surface. A wrong
Euler order throws markers 5-30 cm away, which is trivially separable.

Two candidate compositions of mesh-local point p -> rigid-body-local point:
    "post"  p_rb = R * (S*p) + T          (offset applied in rigid-body frame)
    "pre"   p_rb = R * (S*p + T)          (offset applied in mesh frame)
crossed with every intrinsic/extrinsic Euler sequence and every per-angle sign flip.

Ranking is done on the residual |distance-to-surface - marker_radius|, which is the
quantity that is actually zero for a perfect alignment.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

PROPS_PATH = Path("outputs/motive_props.json")
OUT_PATH = Path("outputs/motive_convention_fit.json")

# Local meshes matching each rigid body's GeometryFile. Only objects whose source mesh
# is available on this machine can be used as evidence.
#
# The profiles reference google_512k/google_64k for some objects but we use google_16k
# throughout: same scanned geometry, coarser tessellation. At 16k the surface sampling
# is still far finer than the centimetre-scale effects being measured here.
MESH_PATHS = {
    "jug": "data/019_pitcher_base/google_16k/textured.obj",
    "drill": "/Users/hxy/Desktop/YCB/035_power_drill/google_16k/textured.obj",
    "crate": "/Users/hxy/Desktop/YCB/align_drill_YCB/Assets/create.obj",
    "table": "/Users/hxy/Desktop/YCB/table.obj",
    "hammer": "/Users/hxy/Desktop/YCB/048_hammer/google_16k/textured.obj",
    "spray": "/Users/hxy/Desktop/YCB/022_windex_bottle/google_16k/textured.obj",
    "football": "/Users/hxy/Desktop/YCB/053_mini_soccer_ball/google_16k/textured.obj",
}

N_SURFACE_SAMPLES = 400_000  # ~0.5 mm spacing on these objects

EULER_SEQS = [
    "".join(p) for p in itertools.permutations("xyz")
] + [
    "".join(p).upper() for p in itertools.permutations("XYZ")
]
SIGNS = list(itertools.product([1, -1], repeat=3))


def angles_for(seq, pitch, yaw, roll):
    """Map pitch/yaw/roll onto the axis letters of `seq` (pitch=X, yaw=Y, roll=Z)."""
    lut = {"x": pitch, "y": yaw, "z": roll}
    return [lut[c.lower()] for c in seq]


def fit_object(name, props, mesh_path):
    geom = props["geometry"]
    S = np.array(geom["scale_factor"])
    T = np.array(geom["offset_m"])
    pitch, yaw, roll = geom["pitch_yaw_roll_deg"]

    markers_rb = np.array([m["offset_m"] for m in props["markers"]])
    radii = np.array([m["diameter_m"] / 2.0 for m in props["markers"]])

    # Scale is applied in the mesh frame, so the remaining mesh->rigid-body step is
    # rigid: we can keep one KD-tree on the scaled surface and inverse-transform the
    # markers into it, which keeps distances exact and every hypothesis cheap.
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.vertices = mesh.vertices * S
    tree = cKDTree(mesh.sample(N_SURFACE_SAMPLES))

    results = []
    for mode in ("post", "pre"):
        for seq in EULER_SEQS:
            for sg in SIGNS:
                ang = np.array(angles_for(seq, pitch, yaw, roll)) * np.array(sg)
                Rm = Rotation.from_euler(seq, ang, degrees=True).as_matrix()
                if mode == "post":  # p_rb = R*p_s + T  ->  p_s = R^T (p_rb - T)
                    p_s = (markers_rb - T) @ Rm
                else:               # p_rb = R*(p_s + T) ->  p_s = R^T p_rb - T
                    p_s = markers_rb @ Rm - T
                d, _ = tree.query(p_s)
                results.append(
                    {
                        "mode": mode,
                        "seq": seq,
                        "signs": list(sg),
                        "residual_mm": float(np.abs(d - radii).mean() * 1000),
                        "max_residual_mm": float(np.abs(d - radii).max() * 1000),
                        "dist_mm": [round(float(v) * 1000, 1) for v in d],
                        "R": Rm.round(6).tolist(),
                    }
                )

    results.sort(key=lambda r: r["residual_mm"])
    return results


def main():
    props = json.loads(PROPS_PATH.read_text())["objects"]
    out = {}
    all_results = {}

    for name, mesh_path in MESH_PATHS.items():
        if not Path(mesh_path).exists():
            print(f"[skip] {name}: mesh not found at {mesh_path}")
            continue
        res = fit_object(name, props[name], mesh_path)
        all_results[name] = res
        best_pre = next(r for r in res if r["mode"] == "pre")

        # Many Euler orders collapse to the same matrix when only one angle is
        # non-trivial, so report distinct rotations rather than distinct spellings.
        distinct, seen = [], []
        for r in res:
            Rm = np.array(r["R"])
            if not any(np.allclose(Rm, s, atol=1e-3) for s in seen):
                seen.append(Rm)
                distinct.append(r)
            if len(distinct) >= 5:
                break

        pyr = props[name]["geometry"]["pitch_yaw_roll_deg"]
        out[name] = {
            "mesh": mesh_path,
            "pitch_yaw_roll_deg": pyr,
            "best": res[0],
            "best_pre_mode": best_pre,
            "distinct_rotations": distinct,
            "n_spellings_tied_with_best": sum(
                1 for r in res if r["residual_mm"] < res[0]["residual_mm"] + 0.1
            ),
        }

        print(f"\n=== {name}  ({props[name]['num_markers']} markers)  pyr={[round(a,2) for a in pyr]} ===")
        print(f"  best overall : {res[0]['mode']:>4} {res[0]['seq']:>4} signs={res[0]['signs']} "
              f"-> mean residual {res[0]['residual_mm']:.1f} mm, max {res[0]['max_residual_mm']:.1f} mm")
        print(f"  best 'pre'   : {best_pre['residual_mm']:.1f} mm  "
              f"(offset-in-mesh-frame hypothesis)")
        print(f"  {out[name]['n_spellings_tied_with_best']} of {len(res)} spellings tie with the best")
        print("  distinct rotation matrices, ranked:")
        for i, r in enumerate(distinct):
            print(f"    {i}: {r['mode']:>4} {r['seq']:>4} signs={str(r['signs']):>12} "
                  f"{r['residual_mm']:>7.1f} mm   dists={r['dist_mm']}")

    # The convention is a property of Motive, not of any one object, so the decisive
    # ranking is over hypotheses that explain *every* object at once.
    joint = {}
    for name, res in all_results.items():
        for r in res:
            key = (r["mode"], r["seq"], tuple(r["signs"]))
            joint.setdefault(key, {})[name] = r["residual_mm"]

    ranked = sorted(
        (
            {
                "mode": k[0],
                "seq": k[1],
                "signs": list(k[2]),
                "per_object_mm": v,
                "mean_mm": float(np.mean(list(v.values()))),
                "worst_mm": float(np.max(list(v.values()))),
            }
            for k, v in joint.items()
            if len(v) == len(all_results)
        ),
        key=lambda r: r["worst_mm"],
    )

    print(f"\n=== joint ranking over {len(all_results)} objects (by worst-object residual) ===")
    print(f"{'mode':>5} {'seq':>4} {'signs':>13} {'worst':>8} {'mean':>8}   per-object")
    for r in ranked[:8]:
        per = "  ".join(f"{n}={v:.1f}" for n, v in r["per_object_mm"].items())
        print(f"{r['mode']:>5} {r['seq']:>4} {str(r['signs']):>13} "
              f"{r['worst_mm']:>7.1f}mm {r['mean_mm']:>7.1f}mm   {per}")

    out["_joint_ranking"] = ranked[:20]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
