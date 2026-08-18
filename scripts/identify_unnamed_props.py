# scripts/identify_unnamed_props.py
"""Identify Captury .skel files whose object name is generic ("prop", "unknown").

Some takes export objects as prop / prop-2 / prop-3 / unknown instead of a real name,
so we cannot tell which rigid body they are by filename. The sorted multiset of
pairwise marker distances is invariant to the frame the markers are expressed in, so it
works as a fingerprint: match each unnamed skel against every Motive rigid body and
report the best fit.

Matching is on distances only, so it is blind to reflection - which is fine here,
handedness is settled separately by fit_captury_motive_alignment.py.
"""
import itertools
import json
import re
from pathlib import Path

import numpy as np

PROPS_PATH = Path("outputs/motive_props.json")
OUT_PATH = Path("outputs/unnamed_prop_identification.json")
DESKTOP = Path("/Users/hxy/Desktop")

SKEL_GLOBS = ["YCB/shot_00*/prop*.skel", "YCB/shot_00*/unknown.skel",
              "YCB/YCB_testdata/Assets/shot_00*/prop*.skel",
              "YCB/YCB_testdata/Assets/shot_00*/unknown.skel"]


def parse_skel(path):
    root, markers = None, []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "Root_tx" and root is None:
            root = np.array([float(v) for v in parts[3:6]])
        if re.match(r"motive_\d+$", parts[0]):
            markers.append([float(v) for v in parts[2:5]])
    return (np.array(markers) - root) / 1000.0 if markers else np.empty((0, 3))


def fingerprint(P):
    return np.sort([np.linalg.norm(P[i] - P[j]) for i, j in itertools.combinations(range(len(P)), 2)])


def main():
    props = json.loads(PROPS_PATH.read_text())["objects"]
    refs = {n: np.array([m["offset_m"] for m in p["markers"]]) for n, p in props.items()}

    out = {}
    paths = sorted({p for g in SKEL_GLOBS for p in DESKTOP.glob(g)})
    for path in paths:
        Q = parse_skel(path)
        if len(Q) < 3:
            continue
        fq = fingerprint(Q)

        scores = []
        for name, P in refs.items():
            if len(P) < len(Q):
                continue
            # Allow the take to be missing markers: try every subset of the profile.
            best = min(
                float(np.abs(fingerprint(P[list(idx)]) - fq).max())
                for idx in itertools.combinations(range(len(P)), len(Q))
            )
            scores.append((best * 1000, name))
        scores.sort()

        rel = str(path).replace(str(DESKTOP) + "/", "")
        out[rel] = {"n_markers": len(Q), "ranking_max_dist_err_mm": scores[:3]}
        top, second = scores[0], scores[1] if len(scores) > 1 else (float("inf"), "-")
        verdict = "confident" if top[0] < 5 and second[0] > 3 * max(top[0], 1) else "AMBIGUOUS"
        print(f"{rel:52s} {len(Q)}mk  -> {top[1]:9s} ({top[0]:6.1f} mm)   "
              f"next {second[1]:9s} ({second[0]:6.1f} mm)   {verdict}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
