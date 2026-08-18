# scripts/probe_person2_anim.py
import ufbx

print("[stage] loading...", flush=True)
scene = ufbx.load_file("data/shot_005/person2.fbx")
print("[stage] loaded, nodes:", len(scene.nodes), flush=True)

print("[stage] anim_stacks count:", len(scene.anim_stacks), flush=True)

stack = scene.anim_stacks[0]
print("[stage] got stack[0]:", str(stack.name), flush=True)

print("[stage] stack.time_begin:", stack.time_begin, flush=True)
print("[stage] stack.time_end:", stack.time_end, flush=True)

print("[stage] stack.layers count:", len(stack.layers), flush=True)

anim = stack.anim
print("[stage] got stack.anim, type:", type(anim), flush=True)

print("[stage] anim.time_begin:", anim.time_begin, flush=True)
print("[stage] anim.time_end:", anim.time_end, flush=True)

print("[stage] ALL DONE", flush=True)