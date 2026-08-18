import ufbx

scene = ufbx.load_file("data/shot_005/jug.fbx")
stack = scene.anim_stacks[0]

print("Anim stack:", stack.name)
print("time:", stack.time_begin, "to", stack.time_end)

print("\nAttributes of anim stack:")
for a in dir(stack):
    if not a.startswith("_"):
        print(a)
