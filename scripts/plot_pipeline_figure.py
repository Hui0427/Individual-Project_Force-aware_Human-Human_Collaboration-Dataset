# scripts/plot_pipeline_figure.py
"""Method overview figure for the report.

Three bands: capture sources on top, the six processing stages in the middle, the
artefact each stage writes at the bottom. Each source is wired to the stage that
actually consumes it. Stage boxes carry a status, so the figure doubles as an honest
progress chart: implemented stages are solid, while the planned joint-fusion output
remains dashed.

Outputs PNG (300 dpi) and PDF (vector, for LaTeX).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent

DONE, PLANNED, SRC, OUT_C, ACC = "#2f6fb0", "#8a8a8a", "#4a4a4a", "#3f7d4f", "#b04a4a"

STAGES = [
    ("1. Capture", True, [
        "2 participants, shared objects",
        "8-12 cam Captury @ 30 fps",
        "Motive marker rigid bodies",
    ], "raw takes\n(.fbx, .skel, video)"),
    ("2. Mesh calibration", True, [
        "recover Motive convention",
        "from 192 candidates",
        "Kabsch: rigid body -> Hips",
        "per-session yaw correction",
    ], "mesh_offsets.json\n(per object x take)"),
    ("3. Per-frame\nreconstruction", True, [
        "object 6-DoF per frame",
        "42 hand joints via FK",
        "all in Captury frame (m)",
    ], "pose csv +\nhand joints csv"),
    ("4. Contact\nannotation", True, [
        "capsule proxy, r = 7-10 mm",
        "signed point-to-mesh",
        "contact label @ 15 mm",
    ], "distances.csv\nsummary.json"),
    ("5. Captury hand\nrefinement", True, [
        "32 joints, anatomical axes",
        "5 losses + 4 robustness gates",
        "FK-consistent, playable",
    ], "refined skeleton\n(.npz, .anim)"),
    ("6. Wuji alignment\n& retargeting", True, [
        "time sync to Captury grid",
        "mirror + wrist calibration",
        "per-frame validity gating",
    ], "joint Captury-Wuji fusion\n(planned)"),
]

# (label, subtitle, index of the stage it feeds)
SOURCES = [
    ("Captury", "body + hand skeletons, object pose", 0),
    ("OptiTrack / Motive", "marker rigid-body definitions", 1),
    ("Object meshes", "YCB scans, Tripo3D", 1),
    ("Wuji glove", "21 joints, 5 EMF, tactile", 5),
]


def box(ax, x, y, w, h, text, color, dashed=False, fs=9, weight="bold", fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.6, edgecolor=color, facecolor=fc,
                                linestyle="--" if dashed else "-", zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight=weight, color=color, zorder=3, linespacing=1.4)


def arrow(ax, p0, p1, color="#555555", dashed=False, rad=0.0, lw=1.4):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                                 linewidth=lw, color=color, zorder=1,
                                 linestyle="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/figures"))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 7.4))
    ax.set_xlim(0, 16); ax.set_ylim(0, 7.4); ax.axis("off")

    bw, gap, x0 = 2.25, 0.34, 0.45
    ystage, hstage = 3.95, 1.05        # stage band
    ydet = 3.70                        # detail bullets start just under the stage box
    yart, hart = 0.95, 0.80            # artefact band
    ysrc, hsrc = 6.15, 0.70            # source band

    def sx(i):                          # centre x of stage i
        return x0 + i * (bw + gap) + bw / 2

    ax.text(8.0, 7.05, "Multimodal human-human-object interaction: processing pipeline",
            ha="center", fontsize=13, fontweight="bold", color="#222222")

    # ---- capture sources, each wired to the stage that consumes it ----
    sw = 3.35
    for i, (name, sub, target) in enumerate(SOURCES):
        x = 0.45 + i * (sw + 0.28)
        box(ax, x, ysrc, sw, hsrc, "", SRC, fc="#fafafa")
        ax.text(x + sw / 2, ysrc + hsrc * 0.62, name, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=SRC, zorder=3)
        ax.text(x + sw / 2, ysrc + hsrc * 0.25, sub, ha="center", va="center",
                fontsize=7.6, style="italic", color="#666666", zorder=3)
        arrow(ax, (x + sw / 2, ysrc), (sx(target), ystage + hstage),
              color="#aaaaaa", rad=0.10 if sx(target) > x + sw / 2 else -0.10,
              dashed=not STAGES[target][1])

    # ---- stages, details, artefacts ----
    for i, (title, done, details, artefact) in enumerate(STAGES):
        x = x0 + i * (bw + gap)
        color = DONE if done else PLANNED
        box(ax, x, ystage, bw, hstage, title, color, dashed=not done, fs=10.5,
            fc="#eef4fa" if done else "#f4f4f4")

        for j, d in enumerate(details):
            ax.text(x + 0.07, ydet - 0.42 - j * 0.245, "– " + d, fontsize=7.8,
                    color="#333333" if done else "#888888", va="top")

        artefact_planned = (i == 5)
        box(ax, x, yart, bw, hart, artefact,
            PLANNED if artefact_planned else OUT_C,
            dashed=artefact_planned, fs=7.8, weight="normal")
        ybul = ydet - 0.42 - len(details) * 0.245
        arrow(ax, (x + bw / 2, ybul - 0.02), (x + bw / 2, yart + hart),
              color="#c8c8c8", dashed=artefact_planned, lw=1.1)

        if i < len(STAGES) - 1:
            arrow(ax, (x + bw, ystage + hstage / 2), (x + bw + gap, ystage + hstage / 2),
                  color="#555555", dashed=not STAGES[i + 1][1])

    # ---- the invariant that holds the pipeline together ----
    xa, xb = sx(1) - bw / 2, sx(4) + bw / 2
    yb = ystage - 0.30
    ax.plot([xa, xa, xb, xb], [yb + 0.12, yb, yb, yb + 0.12], color=ACC, lw=1.3)
    ax.text((xa + xb) / 2, yb - 0.26,
            "object pose and calibration are FIXED downstream — only the hand is optimised",
            ha="center", fontsize=8.8, style="italic", color=ACC)

    ax.text(0.45, 0.35, "solid = implemented      dashed output = planned",
            fontsize=8, color="#666666")
    ax.text(15.55, 0.35,
            "all geometry computed in the Captury right-handed frame (metres); "
            "Unity used for visual verification only",
            fontsize=8, color="#666666", ha="right")

    fig.savefig(out / "fig_pipeline.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig_pipeline.pdf", bbox_inches="tight")
    print(f"wrote {out}/fig_pipeline.png and .pdf")


if __name__ == "__main__":
    main()
