# scripts/hand_refine/extract_local_rotations.py
"""Dump every node's per-frame LOCAL transform (t + quat) from a Captury FBX to .npz,
plus the hierarchy and fixed bone offsets. This is the single source both the hand
optimizer and the Unity .anim exporter read from.

Uses the ufbx access pattern proven safe in extract_person_hand_joints.py
(one-level .parent, persistent node objects, os._exit at the end).

Usage: extract_local_rotations.py <fbx> <out.npz> [fps=60]
"""
import json
import os
import sys

import numpy as np
import ufbx


def main():
    fbx_path, out_path = sys.argv[1], sys.argv[2]
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0

    scene = ufbx.load_file(fbx_path)
    n = len(scene.nodes)
    nodes = [scene.nodes[i] for i in range(n)]
    names = [str(x.name) for x in nodes]
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    parent_idx = []
    for x in nodes:
        p = x.parent
        parent_idx.append(name_to_idx.get(str(p.name), -1) if p is not None else -1)

    stack = scene.anim_stacks[0]
    anim = stack.anim
    t0, t1 = stack.time_begin, stack.time_end
    num_frames = int(round((t1 - t0) * fps)) + 1

    T = np.zeros((num_frames, n, 3))
    Q = np.zeros((num_frames, n, 4))
    for i in range(num_frames):
        t = min(t0 + i / fps, t1)
        for k in range(n):
            tr = ufbx.evaluate_transform(anim, nodes[k], t)
            T[i, k] = (tr.translation.x, tr.translation.y, tr.translation.z)
            Q[i, k] = (tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w)
        if i % 200 == 0:
            print(f"frame {i}/{num_frames}", flush=True)

    np.savez_compressed(
        out_path,
        local_t_mm=T, local_q_xyzw=Q,
        names=np.array(names), parent_idx=np.array(parent_idx),
        fps=fps, time_begin=t0, time_end=t1,
    )
    # Bone-length sanity: local translations must be constant for everything but roots.
    drift = np.abs(T - T[0]).max(axis=0).max(axis=1)
    moving = [names[k] for k in range(n) if drift[k] > 0.01]
    print(f"wrote {out_path}: {num_frames} frames x {n} nodes")
    print(f"nodes with animated local translation (expected: root/Hips only): {moving}")


if __name__ == "__main__":
    main()
    os._exit(0)
