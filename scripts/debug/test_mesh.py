import trimesh

mesh_path = "data/019_pitcher_base/google_16k/textured.obj"
mesh = trimesh.load(mesh_path, force="mesh")

print("Loaded mesh:", mesh_path)
print("Vertices:", mesh.vertices.shape)
print("Faces:", mesh.faces.shape)
print("Bounds:")
print(mesh.bounds)
print("Extents:", mesh.extents)
