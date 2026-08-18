# scripts/hand_refine/export_unity_anim.py
"""Write a Unity AnimationClip (.anim, YAML) from a local-transform npz, so the refined
full-body animation plays directly on the already-imported Captury FBX rig (Generic).

Handedness: the FBX data is right-handed; Unity is left-handed. The importer's
negate-x convention (established for this project as `unity_negate_x`) maps
  position (x,y,z) -> (-x, y, z)      quaternion (x,y,z,w) -> (x, -y, -z, w)
Pass --no-mirror if playback comes out mirrored - same one-bit test as the meshes.

Position curves are written for the Hips only (the single translating node); all other
bones keep their rig-defined offsets. --pos-scale converts the FBX millimetres into
whatever unit the imported rig's Hips localPosition uses in Unity (check once in the
Inspector while the original clip plays; 0.001 if metres, 1 if raw).

Usage: export_unity_anim.py <params.npz> <out.anim> [--clip-name X]
           [--pos-scale 0.001] [--no-mirror]
"""
import argparse

import numpy as np

HEADER = """%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!74 &7400000
AnimationClip:
  m_ObjectHideFlags: 0
  m_CorrespondingSourceObject: {{fileID: 0}}
  m_PrefabInstance: {{fileID: 0}}
  m_PrefabAsset: {{fileID: 0}}
  m_Name: {name}
  serializedVersion: 6
  m_Legacy: 0
  m_Compressed: 0
  m_UseHighQualityCurve: 1
  m_RotationCurves:
"""


def key_q(t, q):
    return (f"      - serializedVersion: 2\n        time: {t:.6f}\n"
            f"        value: {{x: {q[0]:.6f}, y: {q[1]:.6f}, z: {q[2]:.6f}, w: {q[3]:.6f}}}\n"
            f"        inSlope: {{x: 0, y: 0, z: 0, w: 0}}\n"
            f"        outSlope: {{x: 0, y: 0, z: 0, w: 0}}\n        tangentMode: 0\n")


def key_v(t, v):
    return (f"      - serializedVersion: 2\n        time: {t:.6f}\n"
            f"        value: {{x: {v[0]:.6f}, y: {v[1]:.6f}, z: {v[2]:.6f}}}\n"
            f"        inSlope: {{x: 0, y: 0, z: 0}}\n"
            f"        outSlope: {{x: 0, y: 0, z: 0}}\n        tangentMode: 0\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("out")
    ap.add_argument("--clip-name", default="refined_hand")
    ap.add_argument("--pos-scale", type=float, default=0.001)
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args()

    d = np.load(args.npz)
    T, Q = d["local_t_mm"].astype(float), d["local_q_xyzw"].astype(float)
    names = [str(x) for x in d["names"]]
    parent = d["parent_idx"]
    fps = float(d["fps"])
    F, N = Q.shape[:2]

    if not args.no_mirror:
        Q = Q * np.array([1, -1, -1, 1])
        T = T * np.array([-1, 1, 1])

    def path(k):
        parts = []
        while k >= 0:
            if names[k]:
                parts.append(names[k])
            k = parent[k]
        return "/".join(reversed(parts))

    # Export only the skeleton subtree (drop cameras / placeholder mesh nodes).
    root_name = next(n for n in names if n.endswith(":Root"))
    def in_skeleton(k):
        while k >= 0:
            if names[k] == root_name:
                return True
            k = parent[k]
        return False

    # a node is animated if its quaternion moves; static bones get a single key
    qvar = np.abs(Q - Q[0]).max(axis=0).max(axis=1)
    tvar = np.abs(T - T[0]).max(axis=0).max(axis=1)

    with open(args.out, "w") as f:
        f.write(HEADER.format(name=args.clip_name))
        for k in range(N):
            if not names[k] or not in_skeleton(k):
                continue
            f.write("  - curve:\n      serializedVersion: 2\n      m_Curve:\n")
            frames = range(F) if qvar[k] > 1e-6 else [0]
            for i in frames:
                f.write(key_q(i / fps, Q[i, k]))
            f.write("      m_PreInfinity: 2\n      m_PostInfinity: 2\n      m_RotationOrder: 4\n")
            f.write(f"    path: {path(k)}\n")

        f.write("  m_CompressedRotationCurves: []\n  m_EulerCurves: []\n  m_PositionCurves:\n")
        for k in range(N):
            if tvar[k] <= 0.01 or not names[k] or not in_skeleton(k):
                continue
            f.write("  - curve:\n      serializedVersion: 2\n      m_Curve:\n")
            for i in range(F):
                f.write(key_v(i / fps, T[i, k] * args.pos_scale))
            f.write("      m_PreInfinity: 2\n      m_PostInfinity: 2\n")
            f.write(f"    path: {path(k)}\n")

        f.write(f"""  m_ScaleCurves: []
  m_FloatCurves: []
  m_PPtrCurves: []
  m_SampleRate: {fps:g}
  m_WrapMode: 0
  m_Bounds:
    m_Center: {{x: 0, y: 0, z: 0}}
    m_Extent: {{x: 0, y: 0, z: 0}}
  m_ClipBindingConstant:
    genericBindings: []
    pptrCurveMapping: []
  m_AnimationClipSettings:
    serializedVersion: 2
    m_AdditiveReferencePoseClip: {{fileID: 0}}
    m_AdditiveReferencePoseTime: 0
    m_StartTime: 0
    m_StopTime: {(F - 1) / fps:.6f}
    m_OrientationOffsetY: 0
    m_Level: 0
    m_CycleOffset: 0
    m_HasAdditiveReferencePose: 0
    m_LoopTime: 0
    m_LoopBlend: 0
    m_LoopBlendOrientation: 0
    m_LoopBlendPositionY: 0
    m_LoopBlendPositionXZ: 0
    m_KeepOriginalOrientation: 0
    m_KeepOriginalPositionY: 1
    m_KeepOriginalPositionXZ: 0
    m_HeightFromFeet: 0
    m_Mirror: 0
  m_EditorCurves: []
  m_EulerEditorCurves: []
  m_HasGenericRootTransform: 0
  m_HasMotionFloatCurves: 0
  m_Events: []
""")
    print(f"wrote {args.out}  ({F} frames, {N} nodes, fps {fps:g}, "
          f"mirror={'off' if args.no_mirror else 'negate_x'})")


if __name__ == "__main__":
    main()
