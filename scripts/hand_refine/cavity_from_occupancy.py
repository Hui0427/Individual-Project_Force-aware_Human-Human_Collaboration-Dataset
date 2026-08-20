# scripts/hand_refine/cavity_from_occupancy.py
"""Infer which parts of a closed mesh are actually free space, from hand occupancy.

Photogrammetry meshes are watertight shells: Tripo caps the crate's open top and fills
its handle holes, so a hand reaching into a grip reads as "inside solid material". That
fact is absent from the geometry and no geometric test can recover it - rays from the
cavity centre exit through an odd number of faces, exactly like a point in solid stock.

But it IS present in the motion: a region the hand occupies over and over across a take
cannot be solid. We voxelise the object's interior, count how many frames each voxel
holds a hand sample, and treat well-occupied voxels as cavity. Penetration is then only
penalised outside those voxels, so a finger hooked into a grip is left alone while a
finger genuinely pushed into a wall is still corrected.

Deliberately conservative: a voxel must be occupied on at least `min_frames` DISTINCT
frames, so a single mistracked moment cannot carve a hole through the object.
"""
import numpy as np


def cavity_voxels(points_local, inside_mask, voxel_m=0.02, min_frames=20):
    """points_local: (F, S, 3) hand samples in the object frame.
       inside_mask:  (F, S) bool, sample currently reads as inside the mesh.

    Returns (origin, voxel_m, occupancy grid bool[nx,ny,nz]).
    """
    pts = points_local[inside_mask]
    if len(pts) == 0:
        return None
    frame_idx = np.nonzero(inside_mask)[0]

    origin = pts.min(axis=0) - voxel_m
    idx = np.floor((pts - origin) / voxel_m).astype(int)
    dims = idx.max(axis=0) + 2

    # count DISTINCT frames per voxel, not raw samples: a hand resting in one spot
    # would otherwise look identical to a hand sweeping through it once.
    flat = np.ravel_multi_index(idx.T, dims)
    order = np.lexsort((frame_idx, flat))
    flat_s, frame_s = flat[order], frame_idx[order]
    new_cell = np.r_[True, flat_s[1:] != flat_s[:-1]]
    new_frame = np.r_[True, (frame_s[1:] != frame_s[:-1]) | new_cell[1:]]
    counts = np.zeros(int(dims.prod()), dtype=int)
    np.add.at(counts, flat_s[new_frame], 1)

    grid = (counts >= min_frames).reshape(dims)
    return origin, voxel_m, grid


def is_cavity(points_local, cavity):
    """Look points up in the grid returned by cavity_voxels."""
    if cavity is None:
        return np.zeros(points_local.shape[:-1], dtype=bool)
    origin, voxel_m, grid = cavity
    idx = np.floor((points_local - origin) / voxel_m).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(grid.shape)), axis=-1)
    out = np.zeros(points_local.shape[:-1], dtype=bool)
    valid_idx = idx[ok]
    out[ok] = grid[valid_idx[..., 0], valid_idx[..., 1], valid_idx[..., 2]]
    return out
