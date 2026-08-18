#!/usr/bin/env python3
"""把 Wuji 手套 mcap 重采样到 Captury 的帧网格，导出成 Unity 可读的 JSON。

用法:
  python3 wuji_to_unity.py \
      --mcap ego-dell1-2026-08-10/14-48-09-shot_001/wuji/*.mcap \
      --captury captury_postprocess_data/14-48-09-shot_001/meta.txt \
      --out align_drill_YCB/Assets/motion/wuji_1448_A.json

输出的位置是 **Wuji 手腕坐标系** 下的 21 个关键点，已从右手系翻成 Unity 左手系
(x, y, z) -> (x, y, -z)，单位米。手腕的全局位姿不在里面 —— 那个由 Captury 提供。
"""
import argparse, glob, json
from collections import defaultdict

import numpy as np
from mcap.reader import make_reader

JOINT_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
    "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
    "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


def read_skeletons(mcap_path):
    """-> {sn: (side, t[N], pos[N,21,3])}，pos 已转成 Unity 左手系。"""
    raw = defaultdict(list)
    with open(mcap_path, "rb") as f:
        for _, ch, msg in make_reader(f).iter_messages():
            if ch.topic.endswith("/hand_skeleton"):
                raw[ch.topic.split("/")[1]].append(json.loads(msg.data))
    out = {}
    for sn, msgs in raw.items():
        side = "Left" if msgs[0]["header"]["frame_id"].startswith("l_") else "Right"
        names = [j["name"] for j in msgs[0]["joints"]]
        assert names == JOINT_NAMES, f"关键点顺序意外: {names}"
        t = np.array([m["header"]["timestamp_us"] for m in msgs], dtype=np.int64) / 1e6
        pos = np.array([[j["pose"]["position"] for j in m["joints"]] for m in msgs], dtype=np.float64)
        pos[..., 2] *= -1.0                      # 右手系 -> Unity 左手系
        out[sn] = (side, t, pos)
    return out


def read_captury_meta(meta_path):
    meta = {}
    for line in open(meta_path, encoding="utf-8", errors="replace"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            meta[parts[0]] = parts[1]
    num, den = (meta["Framerate"].split("/") + ["1"])[:2]
    return {
        "start_s": int(meta["StartTimestamp"]) / 1e6,
        "fps": float(num) / float(den),
        "frames": int(meta["DurationInFrames"]),
        "name": meta.get("Name", ""),
    }


def resample(t_src, pos_src, t_dst, max_gap):
    """最近邻取样 + 有效性标记（线性插值对手指姿态意义不大，且 120Hz 已远高于 30Hz）。"""
    idx = np.clip(np.searchsorted(t_src, t_dst), 1, len(t_src) - 1)
    left, right = idx - 1, idx
    pick = np.where(np.abs(t_src[left] - t_dst) <= np.abs(t_src[right] - t_dst), left, right)
    err = np.abs(t_src[pick] - t_dst)
    return pos_src[pick], err <= max_gap, err


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcap", required=True)
    ap.add_argument("--captury", required=True, help="Captury 的 meta.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-gap-ms", type=float, default=25.0,
                    help="重采样时允许的最大时间误差，超出即标记为无效帧")
    a = ap.parse_args()

    mcap_path = sorted(glob.glob(a.mcap))[0] if "*" in a.mcap else a.mcap
    cap = read_captury_meta(a.captury)
    hands = read_skeletons(mcap_path)

    t_dst = cap["start_s"] + np.arange(cap["frames"]) / cap["fps"]
    print(f"Captury: {cap['name']}  {cap['frames']} 帧 @ {cap['fps']:.4f} fps"
          f"  [{t_dst[0]:.3f}, {t_dst[-1]:.3f}]")

    doc = {
        "shot": cap["name"], "fps": cap["fps"], "frames": cap["frames"],
        "startTimestampUs": int(cap["start_s"] * 1e6),
        "jointNames": JOINT_NAMES,
        "posLeft": [], "posRight": [], "validLeft": [], "validRight": [],
    }
    for sn, (side, t_src, pos_src) in sorted(hands.items(), key=lambda kv: kv[1][0]):
        pos, valid, err = resample(t_src, pos_src, t_dst, a.max_gap_ms / 1000.0)
        cover = valid.mean() * 100
        print(f"  {side:5s} {sn}  源 {len(t_src)} 帧 [{t_src[0]:.3f},{t_src[-1]:.3f}]"
              f"  -> 覆盖 {valid.sum()}/{cap['frames']} 帧 ({cover:.1f}%)"
              f"  重采样误差 中位 {np.median(err[valid])*1000:.1f} ms")
        doc[f"pos{side}"] = np.round(pos, 5).ravel().tolist()
        doc[f"valid{side}"] = valid.astype(int).tolist()
        doc[f"sn{side}"] = sn

    json.dump(doc, open(a.out, "w"), separators=(",", ":"))
    import os
    print(f"已写入 {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB)")
