import ufbx

scene = ufbx.load_file("data/shot_005/jug.fbx")
anim = scene.anim_stacks[0].anim

root = None
for node in scene.nodes:
    if str(node.name) == "jug:Root":
        root = node
        break

print("root:", root.name)

for t in [0.0, 1.0, 3.0, 5.0, 9.7]:
    tr = ufbx.evaluate_transform(anim, root, t)
    print("\ntime:", t)
    print("translation:", tr.translation)
    print("rotation:", tr.rotation)
    print("scale:", tr.scale)
