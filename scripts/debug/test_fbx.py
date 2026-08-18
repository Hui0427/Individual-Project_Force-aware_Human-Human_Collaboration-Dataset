from pathlib import Path
import ufbx

for f in [
    "data/shot_005/jug.fbx",
    "data/shot_005/person2.fbx",
    "data/shot_005/person3.fbx",
]:
    print("\n====", f, "====")
    scene = ufbx.load_file(f)

    print("nodes:", len(scene.nodes))
    print("meshes:", len(scene.meshes))
    print("anim_stacks:", len(scene.anim_stacks))

    print("\nFirst nodes:")
    for i, node in enumerate(scene.nodes):
        if i >= 20:
            break
        print(i, node.name)