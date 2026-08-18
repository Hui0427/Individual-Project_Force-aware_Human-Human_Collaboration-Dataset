import ufbx
import math

scene = ufbx.load_file("data/shot_005/jug.fbx")
anim = scene.anim_stacks[0].anim

times = [0.0, 1.0, 3.0, 5.0, 9.7]

for node in scene.nodes:
    name = str(node.name)
    if not name:
        continue

    vals = []
    for t in times:
        tr = ufbx.evaluate_transform(anim, node, t)
        vals.append((
            round(tr.translation.x, 6),
            round(tr.translation.y, 6),
            round(tr.translation.z, 6),
            round(tr.rotation.x, 6),
            round(tr.rotation.y, 6),
            round(tr.rotation.z, 6),
            round(tr.rotation.w, 6),
        ))

    moving = len(set(vals)) > 1
    if moving:
        print("\nMOVING NODE:", name)
        for t, v in zip(times, vals):
            print(t, v)
