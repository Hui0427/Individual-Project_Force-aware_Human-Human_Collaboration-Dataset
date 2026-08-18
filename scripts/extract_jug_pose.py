# scripts/extract_jug_pose.py
import csv
import os
import sys
import ufbx

FBX_PATH = "data/shot_005/jug.fbx"
OUT_PATH = "outputs/jug_pose.csv"
NODE_NAME = "jug:Hips"
FPS = 60.0


def main():
    scene = ufbx.load_file(FBX_PATH)
    stack = scene.anim_stacks[0]
    anim = stack.anim
    time_begin = stack.time_begin
    time_end = stack.time_end

    target = None
    for node in scene.nodes:
        if str(node.name) == NODE_NAME:
            target = node
            break
    if target is None:
        raise RuntimeError(f"Could not find node {NODE_NAME}")

    num_frames = int(round((time_end - time_begin) * FPS)) + 1
    os.makedirs("outputs", exist_ok=True)

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time", "tx_mm", "ty_mm", "tz_mm", "qx", "qy", "qz", "qw"])
        for i in range(num_frames):
            t = min(time_begin + i / FPS, time_end)
            tr = ufbx.evaluate_transform(anim, target, t)
            writer.writerow([
                i, round(t, 6),
                tr.translation.x, tr.translation.y, tr.translation.z,
                tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w,
            ])
        f.flush()
        os.fsync(f.fileno())

    print(f"Wrote {num_frames} frames to {OUT_PATH}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
    os._exit(0)  # 跳过 ufbx 原生对象析构，规避退出期 segfault