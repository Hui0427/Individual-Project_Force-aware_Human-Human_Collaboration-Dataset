# scripts/extract_ycb_mesh_info.py
import json
import trimesh

MESH_PATH = "data/019_pitcher_base/google_16k/textured.obj"
OUT_PATH = "outputs/ycb_mesh_info.json"

mesh = trimesh.load(MESH_PATH, force="mesh")

info = {
    "path": MESH_PATH,
    "num_vertices": int(mesh.vertices.shape[0]),
    "num_faces": int(mesh.faces.shape[0]),
    "extents_m": mesh.extents.tolist(),
    "bounds_min_m": mesh.bounds[0].tolist(),
    "bounds_max_m": mesh.bounds[1].tolist(),
    "centroid_m": mesh.centroid.tolist(),
    "assumed_unit": "meters",
}

with open(OUT_PATH, "w") as f:
    json.dump(info, f, indent=2)

print(json.dumps(info, indent=2))