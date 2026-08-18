import ufbx

scene = ufbx.load_file("data/shot_005/jug.fbx")
stack = scene.anim_stacks[0]
anim = stack.anim

print("Anim type:", type(anim))
print("\nAnim attributes:")
for a in dir(anim):
    if not a.startswith("_"):
        print(a)

print("\nModule-level evaluate/search functions:")
for a in dir(ufbx):
    if "eval" in a.lower() or "transform" in a.lower():
        print(a)
