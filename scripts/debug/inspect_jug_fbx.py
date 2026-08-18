import ufbx

f = "data/shot_005/jug.fbx"
scene = ufbx.load_file(f)

print("nodes:", len(scene.nodes))
print("anim_stacks:", len(scene.anim_stacks))

for i, node in enumerate(scene.nodes):
    name = str(node.name)
    print(i, name)

print("\nAnimation stacks:")
for i, stack in enumerate(scene.anim_stacks):
    print(i, stack.name)
    print("time_begin:", stack.time_begin)
    print("time_end:", stack.time_end)
    print("time_duration:", stack.time_duration)
