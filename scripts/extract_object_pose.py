# scripts/extract_object_pose.py
"""Extract an object's per-frame Hips pose from its Captury FBX.

Generalizes extract_jug_pose.py: finds the single `<name>:Hips` node automatically.
Same output format (mm + xyzw quaternion) and the same ufbx access pattern that was
established as segfault-safe in the earlier debugging round.

Usage: extract_object_pose.py <fbx> <out_csv> [fps=60]
"""
import csv
import os
import sys

import ufbx

FPS_DEFAULT = 60.0


def main():
    fbx_path, out_path = sys.argv[1], sys.argv[2]
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else FPS_DEFAULT

    scene = ufbx.load_file(fbx_path)
    stack = scene.anim_stacks[0]
    anim = stack.anim
    time_begin, time_end = stack.time_begin, stack.time_end

    target = None
    for node in scene.nodes:
        if str(node.name).endswith(":Hips"):
            target = node
            break
    if target is None:
        raise RuntimeError("No *:Hips node found")
    print(f"node={target.name}  anim {time_begin:.4f}..{time_end:.4f}s  fps={fps}")

    # NOTE: do not traverse .parent here - the ufbx binding's parent access misbehaves
    # (loops); this was a known hazard from the earlier probing round. Whether this
    # node's local transform equals its world pose is verified downstream by comparing
    # frame 0 against the Captury CSV export, an independent source.
    num_frames = int(round((time_end - time_begin) * fps)) + 1
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time", "tx_mm", "ty_mm", "tz_mm", "qx", "qy", "qz", "qw"])
        for i in range(num_frames):
            t = min(time_begin + i / fps, time_end)
            tr = ufbx.evaluate_transform(anim, target, t)
            writer.writerow([
                i, round(t, 6),
                tr.translation.x, tr.translation.y, tr.translation.z,
                tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w,
            ])
        f.flush()
        os.fsync(f.fileno())

    print(f"Wrote {num_frames} frames to {out_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
    os._exit(0)  # 跳过 ufbx 原生对象析构，规避退出期 segfault
