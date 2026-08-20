#!/usr/bin/env python
# scripts/run_take.py
"""One command per take: extract -> annotate -> refine -> export.

Everything downstream of the (already finished) calibration, driven by a single
takes.yaml entry so batch runs cannot drift from single runs. Each stage is skipped
when its output already exists, so a failed run resumes instead of restarting.

  python scripts/run_take.py --config takes.yaml --take drill_shot005
  python scripts/run_take.py --config takes.yaml --all
  python scripts/run_take.py --config takes.yaml --take X --stages annotate,refine
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

STAGES = ["extract", "annotate", "refine", "export"]


def load_config(path):
    # takes.yaml is written as JSON-with-#-comments so it parses either way.
    text = Path(path).read_text()
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        stripped = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
        return json.loads(stripped)


def sh(cmd, log):
    log.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
    log.flush()
    r = subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"failed ({r.returncode}): {' '.join(str(c) for c in cmd)}")


def run_take(name, cfg, defaults, stages, force):
    t = {**defaults, **cfg}
    data = Path(t["data_dir"])
    out = ROOT / t.get("out_dir", f"outputs/takes/{name}")
    out.mkdir(parents=True, exist_ok=True)
    persons = t.get("persons", ["person1", "person2"])
    obj, fbx = t["object"], t["object_fbx"]
    pose_csv = out / f"{obj}_pose.csv"
    hand_csvs = [out / f"{p}_hand_joints.csv" for p in persons]
    local_npz = [out / f"{p}_locals.npz" for p in persons]

    log_path = out / "run.log"
    print(f"[{name}] -> {out}  (log: {log_path})")
    t0 = time.time()
    with open(log_path, "a") as log:
        log.write(f"\n{'='*70}\n{name}  {time.ctime()}\n{'='*70}\n")

        if "extract" in stages:
            if force or not pose_csv.exists():
                sh([PY, ROOT/"scripts/extract_object_pose.py", data/f"{fbx}.fbx", pose_csv], log)
            for p, hc, lz in zip(persons, hand_csvs, local_npz):
                if force or not hc.exists():
                    sh([PY, ROOT/"scripts/extract_person_hand_joints.py", data/f"{p}.fbx", hc], log)
                if force or not lz.exists():
                    sh([PY, ROOT/"scripts/hand_refine/extract_local_rotations.py",
                        data/f"{p}.fbx", lz], log)

        if "annotate" in stages:
            if force or not (out/"distances.csv").exists():
                cmd = [PY, ROOT/"scripts/compute_hand_object_distances.py",
                       "--object", obj, "--take", t["take_key"], "--mesh", t["mesh"],
                       "--object-pose", pose_csv, "--hands", *hand_csvs,
                       "--out-dir", out, "--threshold-mm", t.get("contact_thr_mm", 15.0)]
                if t.get("sampled") is not None:
                    cmd += ["--sampled", t["sampled"]]
                sh(cmd, log)
                sh([PY, ROOT/"scripts/plot_hoi_results.py", "--out-dir", out,
                    "--mesh", t["mesh"], "--object", obj, "--take", t["take_key"],
                    "--object-pose-name", pose_csv.name], log)

        if "refine" in stages:
            for p, lz in zip(persons, local_npz):
                if p not in t.get("refine_persons", persons):
                    continue
                pout = out / f"refine_{p}"
                if not force and (pout/"hand_params_refined.npz").exists():
                    continue
                fr = t.get("frames") or [0, -1]
                sh([PY, ROOT/"scripts/hand_refine/optimize_hand_pose.py",
                    "--locals", lz, "--person", p, "--object", obj,
                    "--take", t["take_key"], "--mesh", t["mesh"],
                    "--object-pose", pose_csv, "--distances", out/"distances.csv",
                    "--frames", fr[0], fr[1], "--out-dir", pout,
                    "--iters", t.get("iters", 300), "--mode", t.get("mode", "anatomical")], log)

        if "export" in stages:
            for p in t.get("refine_persons", persons):
                pout = out / f"refine_{p}"
                if not (pout/"hand_params_refined.npz").exists():
                    continue
                sh([PY, ROOT/"scripts/hand_refine/export_replayer_json.py",
                    pout/"hand_params_orig.npz", pout/"hand_params_refined.npz",
                    pout/f"replayer_{p}.json"], log)
                sh([PY, ROOT/"scripts/hand_refine/export_unity_anim.py",
                    pout/"hand_params_refined.npz", pout/f"refined_{p}.anim",
                    "--clip-name", f"refined_{p}_{name}"], log)

    print(f"[{name}] done in {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT/"takes.yaml"))
    ap.add_argument("--take")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--force", action="store_true", help="rerun stages even if outputs exist")
    args = ap.parse_args()

    cfg = load_config(args.config)
    defaults = cfg.get("defaults", {})
    takes = cfg["takes"]
    names = list(takes) if args.all else [args.take]
    if not names or names == [None]:
        sys.exit("need --take NAME or --all;  available: " + ", ".join(takes))

    stages = [s for s in args.stages.split(",") if s]
    failed = []
    for n in names:
        try:
            run_take(n, takes[n], defaults, stages, args.force)
        except Exception as e:
            print(f"[{n}] FAILED: {e}")
            failed.append(n)
    if failed:
        sys.exit(f"\n{len(failed)}/{len(names)} takes failed: {failed}")
    print(f"\nall {len(names)} take(s) ok")


if __name__ == "__main__":
    main()
