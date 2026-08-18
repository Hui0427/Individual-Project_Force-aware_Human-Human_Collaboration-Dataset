# scripts/extract_person_hand_joints.py
import csv
import os
import sys
import ufbx

FBX_PATH = "data/shot_005/person2.fbx"
OUT_PATH = "outputs/person2_hand_joints.csv"
FPS = 60.0

HAND_JOINTS = [
    "LeftHand", "RightHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumbEE",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndexEE",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddleEE",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRingEE",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinkyEE",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumbEE",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndexEE",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddleEE",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRingEE",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinkyEE",
]


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    )


def quat_rotate_vec(q, v):
    x, y, z, w = q
    vx, vy, vz = v
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return (rx, ry, rz)


def compose(parent_t, parent_q, local_t, local_q):
    world_q = quat_mul(parent_q, local_q)
    rt = quat_rotate_vec(parent_q, local_t)
    world_t = (parent_t[0] + rt[0], parent_t[1] + rt[1], parent_t[2] + rt[2])
    return world_t, world_q


def main():
    # FPS differs per recording rig: the 12-camera takes are 60 fps, the 8-camera ones
    # (new_data_0716, 0728_data) are 30. Getting this wrong silently doubles or halves
    # the frame count and desynchronises hands from the object track.
    global FPS
    if len(sys.argv) >= 3:
        fbx_path, out_path = sys.argv[1], sys.argv[2]
        if len(sys.argv) >= 4:
            FPS = float(sys.argv[3])
    else:
        fbx_path, out_path = FBX_PATH, OUT_PATH

    print("[stage] loading scene...", flush=True)
    scene = ufbx.load_file(fbx_path)
    node_count = len(scene.nodes)
    print("[stage] scene loaded, node count:", node_count, flush=True)

    # 一次性取出所有节点对象，长期持有，之后再也不重新索引 scene.nodes
    nodes_list = [scene.nodes[i] for i in range(node_count)]
    names_list = [str(n.name) for n in nodes_list]
    name_to_idx = {name: i for i, name in enumerate(names_list)}

    # 一次性建父节点索引表（只访问一次 .parent，全程不再调用）
    parent_idx = []
    for i in range(node_count):
        p = nodes_list[i].parent
        if p is None:
            parent_idx.append(-1)
        else:
            parent_idx.append(name_to_idx.get(str(p.name), -1))
    print("[stage] parent index table built", flush=True)

    stack = scene.anim_stacks[0]
    anim = stack.anim
    time_begin = stack.time_begin
    time_end = stack.time_end
    print("[stage] anim range:", time_begin, time_end, flush=True)

    prefix = None
    for name in names_list:
        if name.endswith(":Hips"):
            prefix = name.split(":Hips")[0] + ":"
            break
    print("[stage] prefix inferred:", prefix, flush=True)
    if prefix is None:
        raise RuntimeError("Could not infer bone name prefix")

    joint_full_names = [prefix + j for j in HAND_JOINTS]
    joint_pairs = [(n, name_to_idx[n]) for n in joint_full_names if n in name_to_idx]

    num_frames = int(round((time_end - time_begin) * FPS)) + 1
    os.makedirs("outputs", exist_ok=True)
    print("[stage] num_frames:", num_frames, "joints found:", len(joint_pairs), flush=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time", "joint", "tx_mm", "ty_mm", "tz_mm", "qx", "qy", "qz", "qw"])
        print("[stage] entering main loop", flush=True)

        for i in range(num_frames):
            if i % 50 == 0:
                print(f"frame {i}/{num_frames}", flush=True)
            t = min(time_begin + i / FPS, time_end)

            # 只用已验证稳定的 evaluate_transform，对持久节点对象逐个取局部变换
            local_cache = [None] * node_count
            for k in range(node_count):
                tr = ufbx.evaluate_transform(anim, nodes_list[k], t)
                local_cache[k] = (
                    (tr.translation.x, tr.translation.y, tr.translation.z),
                    (tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w),
                )

            world_cache = [None] * node_count

            def get_world(k):
                if world_cache[k] is not None:
                    return world_cache[k]
                pk = parent_idx[k]
                lt, lq = local_cache[k]
                if pk == -1:
                    result = (lt, lq)
                else:
                    pt, pq = get_world(pk)
                    result = compose(pt, pq, lt, lq)
                world_cache[k] = result
                return result

            for jname, jidx in joint_pairs:
                wt, wq = get_world(jidx)
                writer.writerow([i, round(t, 6), jname, wt[0], wt[1], wt[2], wq[0], wq[1], wq[2], wq[3]])

        f.flush()
        os.fsync(f.fileno())

    print(f"Wrote {num_frames} frames x {len(joint_pairs)} joints to {out_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
    os._exit(0)