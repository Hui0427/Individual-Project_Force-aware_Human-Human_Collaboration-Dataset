# Force-aware Human–Human Collaboration Dataset

Capture, calibration and annotation pipeline for a multimodal dataset of two
people jointly manipulating objects — with synchronised body motion, hand
skeletons, object pose, and tactile/force sensing.

## Modalities

| Source | What it gives |
|---|---|
| **OptiTrack / Motive** | Marker-based rigid-body pose for each object |
| **Captury** | Markerless body + hand skeletons for both participants |
| **Wuji glove** | 21 hand keypoints + a 24x31 tactile pressure array, recorded as `.mcap` |
| **Object meshes** | YCB scans, plus Tripo3D photogrammetry for non-YCB objects |

Two participants interact with eight objects (jug, drill, table, crate, chair,
football, hammer, spray).

The goal is to place a high-accuracy object mesh (YCB scan or Tripo3D
photogrammetry) at its true pose in **every frame**, then use that to compute
hand–object distances, annotate contact, and refine hand poses against the mesh
surface.

The work splits into two layers that are deliberately kept separate:

| Layer | What it solves | Method | Status |
|---|---|---|---|
| **Calibration** (upstream) | Fixed transform `mesh → Captury Hips`, one constant set per object per session | Closed-form (Kabsch/SVD) + discrete validation — *not* continuous optimisation | Done |
| **Per-frame** (downstream) | Hand joints & object pose per frame → distance, contact labels, penetration, pose refinement | Point-to-mesh / SDF + gradient-based refinement | Working |

## Transform chain

```
mesh --S, R_geo, T_geo (.motive)--> rigid-body pivot
     --R_c, t_c (Kabsch on .skel)--> Captury Hips
     --FBX animation--> world  --negate_x--> Unity
```

Distances are computed entirely in the Captury right-handed frame (metres).
The `negate_x` step exists only for Unity display and does not affect any
distance computation.

![Processing pipeline](outputs/figures/fig_pipeline.png)

## Layout

```
scripts/
  parse_motive_props.py             # .motive  -> motive_props.json
  fit_motive_geometry_convention.py # resolve Motive's geometry convention (one-off)
  identify_unnamed_props.py         # marker-distance fingerprinting (one-off)
  fit_captury_motive_alignment.py   # Kabsch rigid-body -> Hips, per take
  build_mesh_offsets.py             # -> outputs/mesh_offsets.json  (key deliverable)
  auto_align_mesh.py                # attempt at recovering the alignment without the
                                    #   manual Motive step - REJECTED, see Accuracy notes

  extract_object_pose.py            # object FBX  -> per-frame pose CSV
  extract_person_hand_joints.py     # person FBX  -> 42 hand joints/frame (hand-written FK over ufbx)
  compute_hand_object_distances.py  # distances, signed distance, contact labels
  plot_hoi_results.py               # distance curves, contact timeline, 3D closest frame
  run_take.py                       # one command per take: extract -> annotate -> refine -> export

  hand_refine/
    extract_local_rotations.py      # world poses -> per-joint local rotations
    optimize_hand_pose.py           # PyTorch refinement: contact / penetration / deviation / smoothness
    pose_confidence.py              # per-frame, per-joint trust in the Captury hand pose
    object_confidence.py            # object-tracking confidence + hand-proximity gate
    cavity_from_occupancy.py        # infer free space inside closed meshes from hand occupancy
    export_unity_anim.py            # -> .anim for Unity
    export_replayer_json.py         # -> JSON for HandPoseReplayer.cs
    plot_refinement_figures.py      # quantitative before/after paper figures

  wuji/
    wuji_read.py                    # glove .mcap -> npz (21 joints + 24x31 tactile frames)
    wuji_to_unity.py                # resample glove to the Captury frame grid -> Unity JSON

  unity/
    HandPoseReplayer.cs             # A/B playback of original and refined hand poses
    WujiHandReplayer.cs             # retarget Wuji joints onto Captury finger bones
    WujiHandOverlay.cs              # visualise and auto-align the 21 glove keypoints
    ShowAnimatorFrame.cs            # lightweight Unity frame/debug readout

  plot_pipeline_figure.py           # report-ready pipeline PNG and vector PDF

  debug/                            # one-off probes kept for provenance

outputs/
  mesh_offsets.json                 # 8 objects x takes; unity_negate_x and negate_z variants
  captury_motive_alignment.json     # per-take rigid-body -> Hips correction
  motive_props.json, motive_convention_fit.json, ycb_mesh_info.json
  takes/<take>/                     # summary.json, fig1-3 .png, run.log,
                                    #   refine_<person>/skeleton_meta.json
  autoalign_val/                    # evidence that auto_align_mesh.py fails
                                    #   (see Accuracy notes)
  figures/                          # paper figures and quantitative results table

takes.yaml                          # one entry per take; also valid JSON, so pyyaml is optional
HPC.md                              # running the pipeline on a SLURM cluster
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run one take

Edit the paths in `takes.yaml` to point at your local data, then:

```bash
python scripts/run_take.py --config takes.yaml --take drill_shot005
```

Four stages run in order: `extract` (object pose, hand joints, local rotations)
-> `annotate` (distances, contact labels, figures) -> `refine` (hand-pose
optimisation) -> `export` (Unity `.anim` + replayer JSON).

Stages are skipped when their outputs already exist, so an interrupted run
resumes rather than restarting. Use `--all` for every take, `--force` to rerun,
and `--stages refine,export` to run a subset. A failing take does not stop the
others; each gets its own `run.log`.

Individual stages can also be run directly — see [HPC.md](HPC.md) for the
explicit per-script commands and a SLURM job-array template.

## Data

The raw capture data (~33 GB of multi-camera video, FBX, and Motive/Captury
exports) is **not** in this repository. It contains identifiable footage of
study participants and is held under the study's data-management plan.

Object meshes come from the [YCB Object and Model Set](https://www.ycbbenchmarks.com/)
(`google_16k/textured.obj`); non-YCB objects (chair, crate, table) were
reconstructed with Tripo3D.

`Assets_alphabetical.motive` is the Motive asset profile for the capture volume.

`outputs/` keeps the calibration results, per-take `summary.json`, and figures.
The large per-frame CSV/NPZ/anim products are gitignored — regenerate them with
`run_take.py`.

## What constrains the hand refinement

The optimiser moves only the hand subtree, as per-frame axis-angle deltas on 32
joints, and every term is gated by data the optimiser cannot influence:

- **Pose confidence** (`pose_confidence.py`) weights the deviation term per frame
  and per joint. Where Captury is trustworthy the refined pose is held close to
  it; where the tracker clearly failed (folded-back fingers, one-frame pops) the
  anchor is released. It is a weight, not a rejection, so nothing is discarded.
- **Object confidence + proximity gate** (`object_confidence.py`) switch the
  physical terms off when the object's own trajectory teleports, and when the
  hand is nowhere near it. Both are computed from the *original* pose, so the
  optimiser cannot escape a gate by pushing the hand away.
- **Cavity inference** (`cavity_from_occupancy.py`) handles a problem specific to
  photogrammetry meshes: Tripo caps the crate's open top and fills its handle
  holes, so a hand reaching into a grip reads as buried in solid material. No
  geometric test can recover this — but a region the hand occupies repeatedly
  across a take cannot be solid. Well-occupied interior voxels are treated as
  free space, so a finger hooked into a grip is left alone while a finger pushed
  into a wall is still corrected.

The temporal weight defaults to `2e4`, not the original `20`: at `20` the
physical terms were ~4e5 times larger, each frame effectively solved on its own,
and fingertip jerk rose 4-9x above the source animation. A sweep on
crate/shot_012 put the knee at `2e4` — jerk back to the original
0.536 mm/frame^3 with contact unchanged.

## Accuracy notes

- Per-object offset accuracy is roughly **1–3 cm** (chair is worst). Keep this
  error floor in mind when choosing contact thresholds.
- Rigid-body definitions drift between sessions, almost always as pure yaw;
  corrections are per-session and are all recorded in
  `captury_motive_alignment.json`.
- Residuals are used as a **criterion, not an objective**. With 3–6 markers
  against 6 parameters the problem is underdetermined, so free 6-DoF fitting was
  rejected: eight restarts gave solutions 12–27 mm apart while every residual
  looked "better".
- The sampled distance mode (`--sampled 500000`) was measured against the exact
  solver at 0.02 ± 0.03 mm (max 0.22 mm over 500 spot checks) and is ~60x faster.
  `--sampled -1` (the default) now picks per mesh: exact for the ~16k-face YCB
  scans, sampled for the ~2M-face Tripo meshes where exact is intractable.
- **Automatic alignment was attempted and rejected** — a negative result worth
  stating. `auto_align_mesh.py` (stable-pose enumeration + marker fit + support
  plane) reproduced *none* of the five manual alignments it was checked against:
  rotation errors were 84-175 deg, while residuals sometimes looked *better* than
  the manual ones. Hammer is the clean illustration — a 1.4 mm marker residual at
  **174.7 deg** of rotation error. With 3-6 marker constraints against 6 DOF the
  problem is underdetermined, and residual is not a proxy for accuracy. The
  script and its validation outputs are kept as evidence, not as a usable step.

  | object | auto residual | rotation error vs manual |
  |---|---|---|
  | hammer | 1.4 mm | 174.7 deg |
  | crate | 6.2 mm | 84.4 deg |
  | table | 6.2 mm | 87.6 deg |
  | spray | 41.9 mm | 127.8 deg |
  | chair | 45.5 mm | 101.3 deg |

Rigid-body definitions, per-session yaw corrections and the mesh-specific
fixes (chair -90 deg yaw, crate 180 deg about its own vertical axis) are all
recorded in `outputs/mesh_offsets.json` and
`outputs/captury_motive_alignment.json` rather than hard-coded in the scripts.
