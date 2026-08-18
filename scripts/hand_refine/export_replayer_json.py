# scripts/hand_refine/export_replayer_json.py
"""Export original + refined local rotations to one JSON for HandPoseReplayer.cs,
which binds by bone NAME at runtime and therefore cannot suffer the .anim path
binding failures. Only skeleton nodes are included.

Usage: export_replayer_json.py <orig.npz> <refined.npz> <out.json>
"""
import json
import sys

import numpy as np


def main():
    orig, refined, out = sys.argv[1], sys.argv[2], sys.argv[3]
    do = np.load(orig)
    dr = np.load(refined)
    names = [str(x) for x in do["names"]]
    parent = do["parent_idx"]

    root = next(i for i, n in enumerate(names) if n.endswith(":Root"))

    def in_skel(k):
        while k >= 0:
            if k == root:
                return True
            k = parent[k]
        return False

    keep = [k for k in range(len(names)) if names[k] and in_skel(k)]
    hips = next(k for k in keep if names[k].endswith(":Hips"))

    Qo = do["local_q_xyzw"][:, keep].astype(np.float32)
    Qr = dr["local_q_xyzw"][:, keep].astype(np.float32)
    F = Qo.shape[0]

    # Unity's JsonUtility rejects scientific notation (4e-05 etc.), which Python's
    # json module emits for small floats - so serialize numbers in fixed notation.
    def arr(a, dec):
        return "[" + ",".join(f"{v:.{dec}f}" for v in a) + "]"

    def sarr(strings):
        return "[" + ",".join(json.dumps(s) for s in strings) + "]"

    hipsT = do["local_t_mm"][:, hips].astype(np.float32).ravel()
    with open(out, "w") as f:
        f.write('{"fps":%g,"frames":%d,"names":%s,"hipsName":%s,'
                % (float(do["fps"]), F, sarr([names[k] for k in keep]),
                   json.dumps(names[hips])))
        f.write('"qOrig":' + arr(Qo.ravel(), 5) + ",")
        f.write('"q":' + arr(Qr.ravel(), 5) + ",")
        f.write('"hipsT":' + arr(hipsT, 2) + "}")
    print(f"wrote {out}: {F} frames x {len(keep)} bones "
          f"(diff bones: {int((np.abs(Qo-Qr)>1e-6).any(axis=(0,2)).sum())})")


if __name__ == "__main__":
    main()
