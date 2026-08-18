# scripts/inspect_jug_obj.py
import trimesh

m = trimesh.load("data/shot_005/jug.obj", force="mesh")
print("vertices:", m.vertices.shape)
print("faces:", m.faces.shape)
print("extents:", m.extents)
print("bounds:", m.bounds)
print("\nfirst 10 vertices:")
print(m.vertices[:10])