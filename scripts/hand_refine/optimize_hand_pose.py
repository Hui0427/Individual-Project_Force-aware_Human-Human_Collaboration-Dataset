# scripts/hand_refine/optimize_hand_pose.py
"""Refine hand skeleton poses against the object mesh, keeping everything else fixed.

Design (per project plan):
  - Variables: per-frame axis-angle deltas on 32 joints (2 wrists + 2x15 finger joints).
    q_new = q_captury * exp(delta)  ->  always a valid rotation, always playable.
  - FK: forearm world poses are precomputed from the original data (body untouched);
    the hand subtree is re-posed differentiably in PyTorch with fixed bone offsets.
  - Proxy: 3 capsule segments per finger (j1-j2, j2-j3, j3-EE), 3 samples each, with
    per-phalanx radii; distance = signed point-to-surface minus radius. Joint centres
    are never treated as skin.
  - Losses: contact (pull hovering fingertips that the annotation marked as contact
    onto the surface), penetration hinge, deviation from Captury, temporal smoothness,
    soft delta cap standing in for joint limits (per-joint physiological ranges = v2).
  - Correspondence: nearest surface sample refreshed every N iters (ICP-style),
    gradients flow through the fixed correspondence, as in ContactOpt-style refiners.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
RADII_M = {"Thumb": [0.010, 0.009, 0.008], "_default": [0.009, 0.008, 0.007]}


# ---------- quaternion helpers (torch, xyzw) ----------
def qmul(a, b):
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz], -1)


def qrot(q, v):
    qv = q[..., :3]
    t = 2 * torch.cross(qv, v, dim=-1)
    return v + q[..., 3:4] * t + torch.cross(qv, t, dim=-1)


def aa_to_quat(aa):
    ang = aa.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return torch.cat([aa / ang * torch.sin(ang / 2), torch.cos(ang / 2)], -1)


def np_fk_world(T, Q, parent_idx):
    """Full-body FK for all frames (numpy). Returns world t (F,N,3), q (F,N,4)."""
    F, N = T.shape[:2]
    order = sorted(range(N), key=lambda k: 0 if parent_idx[k] < 0 else 1)
    # parents come earlier in ufbx node order; verify then process in index order
    assert all(parent_idx[k] < k for k in range(N)), "nodes not topologically ordered"
    wt = np.zeros_like(T)
    wq = np.zeros_like(Q)
    for k in range(N):
        p = parent_idx[k]
        if p < 0:
            wt[:, k], wq[:, k] = T[:, k], Q[:, k]
        else:
            Rp = Rotation.from_quat(wq[:, p])
            wt[:, k] = wt[:, p] + Rp.apply(T[:, k])
            wq[:, k] = (Rp * Rotation.from_quat(Q[:, k])).as_quat()
    return wt, wq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locals", required=True)
    ap.add_argument("--person", default="person1")
    ap.add_argument("--object", default="drill")
    ap.add_argument("--take", default="shot_005/Drill.skel")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--object-pose", required=True)
    ap.add_argument("--distances", required=True)
    ap.add_argument("--frames", nargs=2, type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--contact-thr-mm", type=float, default=15.0)
    ap.add_argument("--w", nargs=5, type=float, default=[1e4, 2e4, 1.0, 2e4, 10.0],
                    metavar=("CONTACT", "PEN", "DEV", "TEMP", "LIMIT"),
                    help="TEMP was 20 originally, which left the physical terms ~4e5 "
                         "times larger - each frame effectively solved on its own and "
                         "fingertip jerk rose 4-9x above the source animation. A sweep "
                         "on crate/shot_012 (20 / 2e3 / 2e4 / 1e5) put the knee at 2e4: "
                         "jerk back to the original 0.536 mm/frame^3 with contact "
                         "unchanged (-4.38 vs -4.26 mm); 1e5 starts costing contact.")
    ap.add_argument("--mode", choices=["anatomical", "free"], default="anatomical",
                    help="anatomical: finger joints move only along the hinge axes "
                         "recovered from this take's own motion, within the observed "
                         "angle range; wrists capped at --wrist-cap-deg. "
                         "free: v1 behaviour (unconstrained 3-DOF, produces "
                         "anatomically illegal poses - kept for comparison).")
    ap.add_argument("--wrist-cap-deg", type=float, default=15.0)
    ap.add_argument("--range-margin-deg", type=float, default=5.0)
    ap.add_argument("--adaptive-deviation", choices=["on", "off"], default="on",
                    help="on: weight the deviation term by per-frame/per-joint confidence "
                         "in the Captury pose, so implausible tracking (folded fingers, "
                         "one-frame pops) stops anchoring the solution. off: uniform.")
    ap.add_argument("--conf-floor", type=float, default=0.05,
                    help="minimum deviation weight, so no frame is fully unanchored")
    ap.add_argument("--gate-near-mm", type=float, default=20.0,
                    help="physical terms at full strength within this surface distance")
    ap.add_argument("--gate-far-mm", type=float, default=80.0,
                    help="physical terms fade to zero beyond this distance, so a "
                         "mistracked object cannot drag a distant hand around")
    ap.add_argument("--smooth-object", type=int, default=5, metavar="W",
                    help="temporal smoothing window on the object pose used by the "
                         "physical terms (0 = off). These exports are 30 fps sampled at "
                         "60, so the object advances in a staircase while the hands are "
                         "interpolated smooth; without this the terms chatter.")
    ap.add_argument("--pen-cap-mm", type=float, default=25.0,
                    help="Capsule samples that ALREADY sit deeper than this in the input "
                         "pose are excluded from the physical terms entirely (mask fixed "
                         "from the original pose, so it cannot be gamed by moving). "
                         "Photogrammetry meshes are closed shells - Tripo caps the crate's "
                         "open top and its handle holes, verified by ray casting (odd hit "
                         "counts from the cavity centre) and by where the deep samples "
                         "actually sit: clustered at one end, 100%% in the upper half, "
                         "i.e. fingers hooked into a grip the mesh has filled in. Such a "
                         "sample is either in a real cavity or grossly mistracked, and "
                         "pushing it out is wrong either way. Truncating the loss VALUE "
                         "was not enough - the gradient still acted right up to the cap. "
                         "0 disables the mask.")
    ap.add_argument("--cavity-voxel-m", type=float, default=0.02)
    ap.add_argument("--cavity-min-frames", type=int, default=20,
                    help="a voxel counts as cavity only after the hand has been in it on "
                         "this many distinct frames, so one mistracked moment cannot "
                         "carve a hole through the object")
    ap.add_argument("--object-gate", choices=["on", "off"], default="on",
                    help="on: also switch the physical terms off on frames where the "
                         "object's own trajectory looks mistracked (jumps/teleports)")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    f0, f1 = args.frames
    if f1 < 0:
        f1 = None   # resolved after the npz is loaded
    Fseg = None

    d = np.load(args.locals)
    T, Q, names = d["local_t_mm"] / 1000.0, d["local_q_xyzw"], [str(x) for x in d["names"]]
    parent_idx = d["parent_idx"]
    ni = {n: i for i, n in enumerate(names)}
    P = args.person
    # The person and object exports can differ by a frame or two, so every segment is
    # clamped to what both sources actually cover.
    n_obj = sum(1 for _ in csv.DictReader(open(args.object_pose)))
    n_common = min(Q.shape[0], n_obj)
    if f1 is None:
        f1 = n_common
    f1 = min(f1, n_common)
    Fseg = f1 - f0
    print(f"segment: frames {f0}..{f1} ({Fseg} frames)  "
          f"[skeleton {Q.shape[0]}, object {n_obj}]")

    # ---- original world poses (numpy FK) + sanity check against earlier extraction ----
    wt, wq = np_fk_world(T, Q, parent_idx)
    chk = {}
    for r in csv.DictReader(open(Path(args.distances).parent / f"{P}_hand_joints.csv")):
        if int(r["frame"]) == f0 and r["joint"] == f"{P}:LeftHand":
            chk = np.array([float(r["tx_mm"]), float(r["ty_mm"]), float(r["tz_mm"])]) / 1000
            break
    err = np.linalg.norm(wt[f0, ni[f"{P}:LeftHand"]] - chk) * 1000
    print(f"FK sanity vs earlier extraction: LeftHand frame {f0} differs {err:.3f} mm")
    assert err < 1.0, "FK mismatch - abort"

    # ---- hand structure ----
    sides = ["Left", "Right"]
    opt_joints = []          # 32 node indices in optimization order
    for s in sides:
        opt_joints.append(ni[f"{P}:{s}Hand"])
        for fg in FINGERS:
            for lvl in "123":
                opt_joints.append(ni[f"{P}:{s}Hand{fg}{lvl}"])
    ee = {(s, fg): ni[f"{P}:{s}Hand{fg}EE"] for s in sides for fg in FINGERS}
    forearm = {s: ni[f"{P}:{s}ForeArm"] for s in sides}

    # ---- confidence in the Captury pose, computed over the WHOLE take so the robust
    # statistics are not set by whatever happens inside the chosen segment ----
    if args.adaptive_deviation == "on":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from pose_confidence import joint_confidence, summarize
        conf_full, _ = joint_confidence(Q, opt_joints)
        conf_seg = np.clip(conf_full[f0:f1], args.conf_floor, 1.0)
        print(summarize(conf_full, [names[j].split(":", 1)[-1] for j in opt_joints]))
        print(f"deviation weighting: adaptive (segment mean {conf_seg.mean():.3f}, "
              f"min {conf_seg.min():.3f})")
    else:
        conf_seg = np.ones((Fseg, 32))
        print("deviation weighting: uniform")
    conf_t = torch.tensor(conf_seg, dtype=torch.float32)[:, :, None]   # (F,32,1)

    dev = torch.device("cpu")
    q_orig = torch.tensor(Q[f0:f1, opt_joints], dtype=torch.float32)          # (F,32,4)
    off = torch.tensor(T[0, :], dtype=torch.float32)                          # (N,3) fixed
    fa_t = {s: torch.tensor(wt[f0:f1, forearm[s]], dtype=torch.float32) for s in sides}
    fa_q = {s: torch.tensor(wq[f0:f1, forearm[s]], dtype=torch.float32) for s in sides}

    col = {j: c for c, j in enumerate(opt_joints)}
    wrist_cols = [0, 16]
    finger_cols = [c for c in range(32) if c not in wrist_cols]

    # ---- anatomical mode: per-joint hinge axes + angle ranges from the WHOLE take ----
    # Captury's own solver keeps interphalangeal joints on a single axis (measured
    # concentration 98-100%; thumb base ~80%, anatomically 2-DOF), so the axes present
    # in the take ARE the joint's legal axes, and the observed angle range is a
    # subject-specific joint limit.
    if args.mode == "anatomical":
        axes_np = np.zeros((len(finger_cols), 3, 2))
        rng_np = np.zeros((len(finger_cols), 2, 2))       # (joint, axis, [min,max]) rad
        theta_np = np.zeros((Fseg, len(finger_cols), 2))  # original angle along axes
        margin = np.deg2rad(args.range_margin_deg)
        for fi, c in enumerate(finger_cols):
            j = opt_joints[c]
            Rj = Rotation.from_quat(Q[:, j])
            aa = (Rj.mean().inv() * Rj).as_rotvec()
            _, S_, Vt = np.linalg.svd(aa - aa.mean(0), full_matrices=False)
            ndof = 1 if (S_[0] ** 2) / (S_ ** 2).sum() > 0.98 else 2
            for a in range(ndof):
                ax = Vt[a] / np.linalg.norm(Vt[a])
                proj = aa @ ax
                axes_np[fi, :, a] = ax
                rng_np[fi, a] = [proj.min() - margin, proj.max() + margin]
                theta_np[:, fi, a] = proj[f0:f1]
        axes_t = torch.tensor(axes_np, dtype=torch.float32)      # (30,3,2)
        rng_t = torch.tensor(rng_np, dtype=torch.float32)
        theta_t = torch.tensor(theta_np, dtype=torch.float32)
        n2dof = int((axes_np[:, :, 1] != 0).any(axis=1).sum())
        print(f"anatomical mode: {len(finger_cols)-n2dof} hinge joints, {n2dof} 2-DOF joints, "
              f"wrist cap {args.wrist_cap_deg} deg")

    def fk_hands(delta):
        """delta (F,32,3) -> dict node_idx -> world pos (F,3) for hand joints + EEs."""
        qn = qmul(q_orig, aa_to_quat(delta))
        pos, rot = {}, {}
        for s in sides:
            wi = ni[f"{P}:{s}Hand"]
            qw = qmul(fa_q[s], qn[:, col[wi]])
            tw = fa_t[s] + qrot(fa_q[s], off[wi].expand_as(fa_t[s]))
            pos[wi], rot[wi] = tw, qw
            for fg in FINGERS:
                p_idx = wi
                for lvl in "123":
                    ji = ni[f"{P}:{s}Hand{fg}{lvl}"]
                    tj = pos[p_idx] + qrot(rot[p_idx], off[ji].expand_as(tw))
                    qj = qmul(rot[p_idx], qn[:, col[ji]])
                    pos[ji], rot[ji] = tj, qj
                    p_idx = ji
                e = ee[(s, fg)]
                pos[e] = pos[p_idx] + qrot(rot[p_idx], off[e].expand_as(tw))
        return pos, rot

    # ---- capsule sample points ----
    def capsule_points(pos):
        pts, rad, seg_tag = [], [], []
        for s in sides:
            for fg in FINGERS:
                chain = [ni[f"{P}:{s}Hand{fg}{l}"] for l in "123"] + [ee[(s, fg)]]
                radii = RADII_M.get(fg, RADII_M["_default"])
                for si in range(3):
                    a, b = pos[chain[si]], pos[chain[si + 1]]
                    for u in (0.0, 0.5, 1.0):
                        pts.append(a + (b - a) * u)
                        rad.append(radii[si])
                        seg_tag.append((s, fg, si))
        return torch.stack(pts, 1), torch.tensor(rad), seg_tag   # (F,S,3)

    # ---- object frame per frame ----
    op = list(csv.DictReader(open(args.object_pose)))[f0:f1]
    t_o = torch.tensor([[float(r["tx_mm"]), float(r["ty_mm"]), float(r["tz_mm"])] for r in op]) / 1000
    R_o = torch.tensor(Rotation.from_quat(
        [[float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])] for r in op]).as_matrix(),
        dtype=torch.float32)

    offsets = json.loads((Path(__file__).resolve().parents[2] /
                          "outputs/mesh_offsets.json").read_text())[args.object]
    mf = offsets["takes"][args.take]["motive_frame"]
    mesh = trimesh.load(args.mesh, force="mesh")
    mesh.vertices = (mesh.vertices * np.array(offsets["scale"])) @ np.array(mf["R"]).T + np.array(mf["t_m"])
    sp, sf = trimesh.sample.sample_surface(mesh, 200_000, seed=0)
    tree = cKDTree(sp)
    sp_t = torch.tensor(sp, dtype=torch.float32)
    sn_t = torch.tensor(mesh.face_normals[sf], dtype=torch.float32)

    # ---- gates: the object may only influence the hand when it is (a) tracked
    # plausibly and (b) actually near it. Both are computed from input data, never
    # from the optimiser state, so they cannot be gamed by pushing the hand away. ----
    from object_confidence import (object_confidence, proximity_gate,
                                   summarize as osum, detect_duplicate_frames,
                                   smooth_object_pose)
    dup = detect_duplicate_frames(t_o.numpy())
    if args.smooth_object > 0:
        t_sm, R_sm = smooth_object_pose(t_o.numpy(), Rotation.from_matrix(R_o.numpy()),
                                        args.smooth_object)
        moved = np.linalg.norm(t_sm - t_o.numpy(), axis=1).max() * 1000
        print(f"object pose: {dup*100:.0f}% duplicate frames (30fps sampled at 60); "
              f"smoothing w={args.smooth_object} moved it by at most {moved:.1f} mm")
        t_o = torch.tensor(t_sm, dtype=torch.float32)
        R_o = torch.tensor(R_sm.as_matrix(), dtype=torch.float32)
    else:
        print(f"object pose: {dup*100:.0f}% duplicate frames, smoothing OFF")
    if args.object_gate == "on":
        obj_conf = object_confidence(t_o.numpy(), Rotation.from_matrix(R_o.numpy()))
    else:
        obj_conf = np.ones(Fseg)

    min_d = np.full(Fseg, 1e9)
    for r in csv.DictReader(open(args.distances)):
        f = int(r["frame"])
        if r["person"] != P or not (f0 <= f < f1):
            continue
        min_d[f - f0] = min(min_d[f - f0], float(r["signed_dist_mm"]))
    min_d[min_d > 1e8] = 1e9
    gate = proximity_gate(min_d, args.gate_near_mm, args.gate_far_mm)
    print(osum(obj_conf, gate))
    w_phys = torch.tensor(obj_conf * gate, dtype=torch.float32)[:, None]   # (F,1)

    # Per-sample mask: freeze whatever was already deep inside the (closed) mesh.
    with torch.no_grad():
        pts0, rad0, _ = capsule_points(fk_hands(torch.zeros(Fseg, 32, 3))[0])
        loc0 = torch.einsum("fij,fsj->fsi", R_o.transpose(1, 2), pts0 - t_o[:, None, :])
        d0, i0 = tree.query(loc0.reshape(-1, 3).numpy())
        s0 = np.sign(np.einsum("ij,ij->i", loc0.reshape(-1, 3).numpy() - sp[i0], mesh.face_normals[sf][i0]))
        signed0 = (s0 * d0).reshape(Fseg, -1) * 1000 - rad0.numpy() * 1000
    if args.pen_cap_mm > 0:
        # Which interior regions are actually free space is inferred from where the hand
        # repeatedly goes (see cavity_from_occupancy). Masking by depth alone fails: at
        # sample level the joints drag frozen points along anyway, and at finger level it
        # mutes exactly the fingers doing the gripping.
        from cavity_from_occupancy import cavity_voxels, is_cavity
        inside0 = signed0 <= -args.pen_cap_mm
        cav = cavity_voxels(loc0.numpy(), inside0, args.cavity_voxel_m,
                            args.cavity_min_frames)
        in_cav = is_cavity(loc0.numpy(), cav)
        n_vox = int(cav[2].sum()) if cav else 0
        print(f"cavity from occupancy: {n_vox} voxels ({args.cavity_voxel_m*100:g} cm, "
              f"seen on >={args.cavity_min_frames} frames) covering "
              f"{in_cav.mean()*100:.1f}% of samples")
    else:
        cav = None
    valid = torch.ones(Fseg, signed0.shape[1])

    # ---- contact targets from the annotation ----
    thr = args.contact_thr_mm
    contact = {}   # (frame_local, side, finger) -> True
    for r in csv.DictReader(open(args.distances)):
        f = int(r["frame"])
        if r["person"] != P or not (f0 <= f < f1) or not r["joint"].endswith("EE"):
            continue
        if float(r["signed_dist_mm"]) < thr:
            side = "Left" if r["joint"].startswith("Left") else "Right"
            fg = r["joint"].replace(f"{side}Hand", "").replace("EE", "")
            contact[(f - f0, side, fg)] = True
    print(f"contact targets in segment: {len(contact)} (frame,finger) pairs")

    wC, wP, wD, wT, wL = args.w
    if args.mode == "anatomical":
        dtheta = torch.zeros(Fseg, len(finger_cols), 2, requires_grad=True)
        dwrist = torch.zeros(Fseg, 2, 3, requires_grad=True)
        params = [dtheta, dwrist]
        wrist_cap = float(np.deg2rad(args.wrist_cap_deg))

        fcols_t = torch.tensor(finger_cols)
        wcols_t = torch.tensor(wrist_cols)
        frows = torch.arange(Fseg)[:, None]

        def build_delta():
            fa = torch.einsum("jca,fja->fjc", axes_t, dtheta)   # (F,30,3) axis-angle
            d = torch.zeros(Fseg, 32, 3)
            d = d.index_put((frows, fcols_t[None, :]), fa)
            d = d.index_put((frows, wcols_t[None, :]), dwrist)
            return d

        def limit_loss(delta):
            th = theta_t + dtheta
            over = (th - rng_t[None, :, :, 1]).clamp(min=0) + (rng_t[None, :, :, 0] - th).clamp(min=0)
            wr = (dwrist.norm(dim=-1) - wrist_cap).clamp(min=0)
            return (over ** 2).mean() * 100 + (wr ** 2).mean() * 100
    else:
        free_delta = torch.zeros(Fseg, 32, 3, requires_grad=True)
        params = [free_delta]
        cap = float(np.deg2rad(45.0))

        def build_delta():
            return free_delta

        def limit_loss(delta):
            return ((delta.norm(dim=-1) - cap).clamp(min=0) ** 2).mean()

    opt = torch.optim.Adam(params, lr=0.01)

    nn_idx = None
    log = []
    for it in range(args.iters):
        delta = build_delta()
        pos, _ = fk_hands(delta)
        pts, rad, tags = capsule_points(pos)                     # (F,S,3)
        loc = torch.einsum("fij,fsj->fsi", R_o.transpose(1, 2), pts - t_o[:, None, :])

        if it % 25 == 0:
            with torch.no_grad():
                _, nn_idx = tree.query(loc.reshape(-1, 3).numpy())
                nn_idx = torch.tensor(nn_idx)
        s_pts = sp_t[nn_idx].reshape(loc.shape)
        s_nrm = sn_t[nn_idx].reshape(loc.shape)
        vec = loc - s_pts
        sgn = (vec * s_nrm).sum(-1).sign()
        if cav is not None:
            # A sample inside an inferred cavity is in FREE SPACE touching the inner
            # wall, not buried in material: force its distance positive so the field is
            # physically correct. Fixing the field beats masking the loss - a mask leaks
            # through the kinematic chain (joints are shared by every sample on a finger),
            # which is why sample-, finger- and voxel-level masks all failed to hold.
            with torch.no_grad():
                inc = torch.tensor(is_cavity(loc.detach().numpy(), cav))
            sgn = torch.where(inc, torch.ones_like(sgn), sgn)
        signed = sgn * vec.norm(dim=-1) - rad   # (F,S)

        cmask = torch.zeros_like(signed, dtype=torch.bool)
        for si, (s, fg, lvl) in enumerate(tags):
            if lvl == 2:   # distal segment
                for f in range(Fseg):
                    if contact.get((f, s, fg)):
                        cmask[f, si] = True

        gw = w_phys.expand_as(signed) * valid
        L_contact = ((gw * signed.clamp(min=0) ** 2)[cmask].mean()
                     if cmask.any() else signed.sum() * 0)
        L_pen = (gw * signed.clamp(max=0) ** 2).mean()
        L_dev = (conf_t * delta ** 2).mean()   # low confidence -> weak anchor
        L_temp = ((delta[1:] - delta[:-1]) ** 2).mean()
        L_lim = limit_loss(delta)
        loss = wC * L_contact + wP * L_pen + wD * L_dev + wT * L_temp + wL * L_lim

        opt.zero_grad(); loss.backward(); opt.step()
        if it % 50 == 0 or it == args.iters - 1:
            row = dict(it=it, loss=float(loss), contact=float(L_contact), pen=float(L_pen),
                       dev=float(L_dev), temp=float(L_temp), lim=float(L_lim))
            log.append(row)
            print("  " + "  ".join(f"{k}={v:.3e}" if k != "it" else f"it={v}" for k, v in row.items()))

    # ---- metrics + save ----
    delta = build_delta().detach()
    with torch.no_grad():
        for tag, dlt in (("before", torch.zeros_like(delta)), ("after", delta)):
            pos, _ = fk_hands(dlt)
            pts, rad, tags = capsule_points(pos)
            loc = torch.einsum("fij,fsj->fsi", R_o.transpose(1, 2), pts - t_o[:, None, :])
            dd, ii = tree.query(loc.reshape(-1, 3).numpy())
            vec = loc.reshape(-1, 3).numpy() - sp[ii]
            sgn = np.sign((vec * mesh.face_normals[sf][ii]).sum(-1))
            sd = (sgn * dd - rad.numpy().repeat(Fseg).reshape(-1, Fseg).T.reshape(-1)) * 1000
            sd2 = (sgn * dd) * 1000 - np.tile(rad.numpy(), Fseg) * 1000
            sd = sd2.reshape(Fseg, -1)
            cm = cmask.numpy()
            print(f"{tag:6s}: contact 段 capsule 距离 {sd[cm].mean():7.2f} mm | "
                  f"最深穿透 {sd.min():7.2f} mm | 穿透样本占比 {(sd<0).mean()*100:.1f}%")
            if tag == "after":
                metrics_after = dict(contact_mean_mm=float(sd[cm].mean()),
                                     worst_pen_mm=float(sd.min()),
                                     pen_frac=float((sd < 0).mean()))

        qn = qmul(q_orig, aa_to_quat(delta)).numpy()
        Q_ref = Q.copy()
        Q_ref[f0:f1, opt_joints] = qn
        dq = np.rad2deg(np.linalg.norm(delta.numpy(), axis=-1))
        np.savez_compressed(out / "hand_params_refined.npz",
                            local_q_xyzw=Q_ref, local_t_mm=d["local_t_mm"],
                            names=d["names"], parent_idx=parent_idx, fps=d["fps"],
                            opt_joint_indices=np.array(opt_joints),
                            frames=np.array([f0, f1]),
                            delta_aa=delta.numpy())
        meta = dict(person=P, object=args.object, take=args.take, frames=[f0, f1],
                    fps=float(d["fps"]), opt_joints=[names[j] for j in opt_joints],
                    bone_lengths_mm={names[k]: float(np.linalg.norm(d["local_t_mm"][0, k]))
                                     for k in opt_joints},
                    weights=dict(contact=wC, pen=wP, dev=wD, temp=wT, limit=wL),
                    delta_deg=dict(mean=float(dq.mean()), max=float(dq.max())),
                    metrics_after=metrics_after, log=log)
        (out / "skeleton_meta.json").write_text(json.dumps(meta, indent=2))
        np.savez_compressed(out / "hand_params_orig.npz",
                            local_q_xyzw=Q, local_t_mm=d["local_t_mm"],
                            names=d["names"], parent_idx=parent_idx, fps=d["fps"])
    print(f"平均改动 {dq.mean():.2f}°, 最大 {dq.max():.2f}°")
    print(f"wrote {out}/hand_params_refined.npz, skeleton_meta.json")


if __name__ == "__main__":
    main()
