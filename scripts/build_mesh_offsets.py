# scripts/build_mesh_offsets.py
"""Compose the full mesh -> Captury-Hips transform for each object, ready for Unity.

Chain (all in Motive's right-handed Y-up frame, metres):

    mesh vertex  --scale S-->  --R_geo, T_geo (from .motive)-->  rigid-body pivot
                 --R_rb2cap, t_rb2cap (Kabsch on the take's .skel)-->  Captury Hips

    p_hips = R_rb2cap @ (R_geo @ (S * p_mesh) + T_geo) + t_rb2cap
           = (R_rb2cap @ R_geo) @ (S * p_mesh) + (R_rb2cap @ T_geo + t_rb2cap)

R_geo uses the convention recovered by fit_motive_geometry_convention.py:
    R_geo = Rx(pitch) @ Ry(yaw) @ Rz(roll), angles exactly as stored, offset applied
    after the rotation, GeometryScale as a percentage.

Unity's local matrix is T*R*S, which matches this form exactly, so GeometryScale maps
straight onto localScale with no residual shear - even where it is non-uniform
(table 160/166/160, chair 80/85/80).

Unity is left-handed, so the result is additionally reported under a basis change
M = diag(-1,1,1) (and the diag(1,1,-1) alternative): R_u = M R M, t_u = M t, and the
scale is unchanged because M is diagonal and commutes with it.
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

PROPS_PATH = Path("outputs/motive_props.json")
ALIGN_PATH = Path("outputs/captury_motive_alignment.json")
OUT_PATH = Path("outputs/mesh_offsets.json")

BASIS_CANDIDATES = {"negate_x": np.diag([-1.0, 1.0, 1.0]), "negate_z": np.diag([1.0, 1.0, -1.0])}

# Yaw corrections needed to make a profile match the LOCAL copy of its mesh.
#
# Important: this corrects a mismatch between the stored alignment and the mesh file on
# this machine. It does NOT establish that the Motive profile is wrong - the profile
# references a Windows path we cannot read, and a Tripo mesh re-generated from the same
# photos comes back with a different arbitrary heading. So "profile is wrong" and "local
# mesh is a different export" are indistinguishable from here, and the correction is a
# property of the mesh file, not of the profile.
#
#   chair: local chair.obj (md5 6ec1d5f5e7ccc9019d952e12b8a63564, a renamed copy of
#          tripo_convert_05a74504-...obj) scores 50.1 mm at the stored 34.8 deg and
#          7.4 mm at -90. Independently confirmed in Unity by the user, who saw the seat
#          facing wrong and fixed it with the same +90 (sign flips across handedness).
#          Re-check this if the chair mesh is ever swapped or re-generated.
#          Note crate and table do NOT show this mismatch, so their local copies do
#          agree with what Motive used.
#
# NOT applied, deliberately:
#   table: +180 improves 8.4 -> 6.9 mm, but the table is the most 180-deg-symmetric
#          object measured (0.2% self-match), so the test is nearly blind here and
#          1.5 mm proves nothing. Needs a visual check against an asymmetric feature.
#   crate: its error is a 180-deg flip about the MESH's own vertical axis, which is a
#          different family from a yaw applied mid-chain (its pitch/roll are non-zero,
#          so the two do not commute). Markers cannot see it at all; it was found from
#          the direction of travel. Left out until confirmed in Unity.
GEOMETRY_YAW_CORRECTION_DEG = {"chair": -90.0}


def geometry_rotation(pitch, yaw, roll):
    """Motive's GeometryPitchYawRoll -> matrix, per the empirically recovered convention."""
    return Rotation.from_euler("XYZ", [pitch, yaw, roll], degrees=True).as_matrix()


def main():
    props = json.loads(PROPS_PATH.read_text())["objects"]
    align = json.loads(ALIGN_PATH.read_text())
    out = {}

    for name, p in sorted(props.items()):
        geom = p["geometry"]
        if geom is None:
            continue

        S = np.array(geom["scale_factor"])
        pitch, yaw, roll = geom["pitch_yaw_roll_deg"]
        yaw_fix = GEOMETRY_YAW_CORRECTION_DEG.get(name, 0.0)
        R_geo = geometry_rotation(pitch, yaw + yaw_fix, roll)
        T_geo = np.array(geom["offset_m"])

        entry = {"mesh": geom["file_basename"], "scale": S.tolist(), "takes": {}}
        if yaw_fix:
            entry["yaw_correction_deg"] = yaw_fix
            entry["yaw_correction_note"] = (
                "the alignment stored in the .motive profile is wrong by this much; "
                "corrected here after verifying against the markers"
            )

        # The rigid body drifts between sessions, so there is no single offset per
        # object - one per take. Objects with no .skel get a lone "assumed" entry.
        takes = align.get(name) or {
            "ASSUMED_identity": {
                "R_motive_to_captury": np.eye(3).tolist(),
                "t_motive_to_captury_m": [0.0, 0.0, 0.0],
                "rms_mm": None,
                "angle_deg": 0.0,
                "warning": "no .skel anywhere on this machine; assumes Hips frame == "
                           "rigid-body frame, which is FALSE for every object measured "
                           "so far except table/drill/crate in their own session",
            }
        }

        for take, a in takes.items():
            R_c = np.array(a["R_motive_to_captury"])
            t_c = np.array(a["t_motive_to_captury_m"])

            R_tot = R_c @ R_geo
            t_tot = R_c @ T_geo + t_c

            t_entry = {
                "skel_fit_rms_mm": a["rms_mm"],
                "rb_to_captury_angle_deg": a["angle_deg"],
                "motive_frame": {
                    "R": R_tot.round(6).tolist(),
                    "euler_XYZ_deg": Rotation.from_matrix(R_tot)
                    .as_euler("XYZ", degrees=True).round(4).tolist(),
                    "t_m": t_tot.round(6).tolist(),
                },
            }
            if "warning" in a:
                t_entry["warning"] = a["warning"]

            for bname, M in BASIS_CANDIDATES.items():
                R_u = M @ R_tot @ M
                t_u = M @ t_tot
                # Unity reads out local rotation as ZXY-order Euler angles.
                eul = Rotation.from_matrix(R_u).as_euler("ZXY", degrees=True)
                t_entry[f"unity_{bname}"] = {
                    "localPosition": t_u.round(5).tolist(),
                    "localEulerAngles_ZXY": [round(float(a_), 3) for a_ in (eul[1], eul[2], eul[0])],
                    "localScale": S.tolist(),
                }

            entry["takes"][take] = t_entry

        out[name] = entry

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))

    print(f"{'object':9s} {'take':22s} {'rb->hips':>9s} {'localPosition (m)':>26s}   "
          f"{'localEuler XYZ (deg)':>25s}")
    for name, e in out.items():
        for take, t in e["takes"].items():
            u = t["unity_negate_x"]
            flag = "  <-- ASSUMED" if "warning" in t else ""
            print(f"{name:9s} {take[:22]:22s} {t['rb_to_captury_angle_deg']:>8.1f}° "
                  f"{str([round(v,4) for v in u['localPosition']]):>26s}   "
                  f"{str([round(v,1) for v in u['localEulerAngles_ZXY']]):>25s}{flag}")
    print(f"\nwrote {OUT_PATH}  (both handedness candidates included)")


if __name__ == "__main__":
    main()
