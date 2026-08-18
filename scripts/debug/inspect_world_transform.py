# scripts/inspect_world_transform.py
import ufbx

scene = ufbx.load_file("data/shot_005/person2.fbx")
anim = scene.anim_stacks[0].anim

baked = ufbx.evaluate_scene(scene, anim, 1.0)

target = None
for node in baked.nodes:
    if str(node.name) == "person2:LeftHandIndex1":
        target = node
        break

print("node found:", target is not None)
print("\nattributes on baked node:")
for a in dir(target):
    if not a.startswith("_"):
        print(a)