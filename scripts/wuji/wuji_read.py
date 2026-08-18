#!/usr/bin/env python3
"""读取 Wuji 手套录制的 .mcap，导出成便于分析的 numpy 数组 / npz。

用法:
    python3 wuji_read.py ego-dell1-2026-08-10/14-48-09-shot_001/wuji/*.mcap [-o out.npz]

依赖: pip install mcap numpy
"""
import argparse
import json
from collections import defaultdict

import numpy as np
from mcap.reader import make_reader

# hand_skeleton 的 21 个关键点顺序（MediaPipe 命名）
JOINT_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
    "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
    "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]
TACTILE_ROWS, TACTILE_COLS = 24, 31


def load(path):
    """返回 {device_sn: {stream: dict of arrays}}，时间戳统一用设备时间(秒, epoch)。"""
    raw = defaultdict(lambda: defaultdict(list))
    with open(path, "rb") as f:
        for _, ch, msg in make_reader(f).iter_messages():
            _, sn, stream = ch.topic.split("/", 2)
            raw[sn][stream].append(json.loads(msg.data))

    out = {}
    for sn, streams in raw.items():
        dev = {}
        for stream, msgs in streams.items():
            t = np.array([m["header"]["timestamp_us"] for m in msgs], dtype=np.int64) / 1e6
            seq = np.array([m["header"]["seq"] for m in msgs], dtype=np.int64)
            frame_id = msgs[0]["header"]["frame_id"]
            d = {"t": t, "seq": seq, "frame_id": frame_id}

            if stream == "hand_skeleton":
                # (N, 21, 3) 位置 / (N, 21, 4) 四元数 xyzw / (N, 21) 置信度
                d["pos"] = np.array([[j["pose"]["position"] for j in m["joints"]] for m in msgs], dtype=np.float32)
                d["quat_xyzw"] = np.array(
                    [[[j["pose"]["orientation"][k] for k in "xyzw"] for j in m["joints"]] for m in msgs],
                    dtype=np.float32)
                d["conf"] = np.array([[j["confidence"] for j in m["joints"]] for m in msgs], dtype=np.float32)
                d["names"] = [j["name"] for j in msgs[0]["joints"]]
            elif stream == "emf_poses":
                # (N, 5, 3) 指尖线圈位置（拇指→小指），在 {l,r}_hand_emf_tx 系下
                d["pos"] = np.array([[p["pose"]["position"] for p in m["poses"]] for m in msgs], dtype=np.float32)
                d["quat_xyzw"] = np.array(
                    [[[p["pose"]["orientation"][k] for k in "xyzw"] for p in m["poses"]] for m in msgs],
                    dtype=np.float32)
                d["conf"] = np.array([[p["confidence"] for p in m["poses"]] for m in msgs], dtype=np.float32)
            elif stream == "tactile":
                # (N, 24, 31)，-1 表示该点无传感器（无效）
                a = np.array([m["data"] for m in msgs], dtype=np.float32)
                d["matrix"] = a.reshape(-1, TACTILE_ROWS, TACTILE_COLS)
                d["valid_mask"] = d["matrix"][0] >= 0
            elif stream == "tactile_zones":
                for zone in ("palm", "thumb", "index", "middle", "ring", "pinky"):
                    d[zone] = np.array([m[zone] for m in msgs], dtype=np.float32)
            dev[stream] = d
        out[sn] = dev
    return out


def describe(data):
    for sn, dev in sorted(data.items()):
        side = dev["hand_skeleton"]["frame_id"] if "hand_skeleton" in dev else "?"
        print(f"\n=== {sn}  ({side}) ===")
        for stream, d in sorted(dev.items()):
            t = d["t"]
            hz = len(t) / (t[-1] - t[0]) if len(t) > 1 else float("nan")
            print(f"  {stream:14s} n={len(t):5d}  {hz:6.1f} Hz  t=[{t[0]:.3f}, {t[-1]:.3f}]")
        if "hand_skeleton" in dev:
            p = dev["hand_skeleton"]["pos"]
            thumb, index = JOINT_NAMES.index("thumb_tip"), JOINT_NAMES.index("index_finger_tip")
            pinch = np.linalg.norm(p[:, thumb] - p[:, index], axis=-1)
            print(f"  拇指-食指指尖距离: {pinch.min()*100:.1f} ~ {pinch.max()*100:.1f} cm")
        if "tactile" in dev:
            m = dev["tactile"]["matrix"]
            mask = dev["tactile"]["valid_mask"]
            print(f"  触觉有效点 {mask.sum()}/{mask.size}，压力峰值 {m[:, mask].max():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mcap")
    ap.add_argument("-o", "--out", help="导出 npz 路径")
    args = ap.parse_args()

    data = load(args.mcap)
    describe(data)

    if args.out:
        flat = {}
        for sn, dev in data.items():
            for stream, d in dev.items():
                for k, v in d.items():
                    if isinstance(v, np.ndarray):
                        flat[f"{sn}/{stream}/{k}"] = v
        np.savez_compressed(args.out, **flat)
        print(f"\n已写入 {args.out}（{len(flat)} 个数组）")
