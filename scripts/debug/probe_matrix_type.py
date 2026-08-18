# scripts/probe_matrix_type.py
import ufbx

scene = ufbx.load_file("data/shot_005/person2.fbx")
anim = scene.anim_stacks[0].anim

baked = ufbx.evaluate_scene(scene, anim, 1.0)
node_count = len(baked.nodes)

target = None
for i in range(node_count):
    n = baked.nodes[i]
    if str(n.name) == "person2:LeftHand":
        target = n
        break

m = target.node_to_world
print("type of matrix:", type(m))
print("\nattributes/methods on matrix:")
for a in dir(m):
    if not a.startswith("_"):
        print(a)

print("\nrepr:", m)