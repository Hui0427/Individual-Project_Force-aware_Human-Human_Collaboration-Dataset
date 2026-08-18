# scripts/probe_fbx_nodes.py
import subprocess
import sys

if len(sys.argv) != 2:
    print("Usage: python scripts/probe_fbx_nodes.py <path_to_fbx>")
    sys.exit(1)

FBX_PATH = sys.argv[1]

count_code = f"""
import ufbx
scene = ufbx.load_file({FBX_PATH!r})
print(len(scene.nodes))
"""
r = subprocess.run([sys.executable, "-c", count_code], capture_output=True, text=True)
if r.returncode != 0:
    print("[CRASH] even loading the scene failed:", r.stderr.strip()[-300:])
    sys.exit(1)

n = int(r.stdout.strip())
print("total nodes:", n)

for i in range(n):
    node_code = f"""
import ufbx
scene = ufbx.load_file({FBX_PATH!r})
node = scene.nodes[{i}]
print({i}, repr(str(node.name)))
"""
    r = subprocess.run([sys.executable, "-c", node_code], capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        print(f"[CRASH] node {i}: returncode={r.returncode} stderr={r.stderr.strip()[-200:]}")
    else:
        print(r.stdout.strip())

# import subprocess
# import sys

# FBX_PATH = "data/shot_005/person2.fbx"

# count_code = f"""
# import ufbx
# scene = ufbx.load_file({FBX_PATH!r})
# print(len(scene.nodes))
# """
# r = subprocess.run([sys.executable, "-c", count_code], capture_output=True, text=True)
# n = int(r.stdout.strip())
# print("total nodes:", n)

# for i in range(n):
#     node_code = f"""
# import ufbx
# scene = ufbx.load_file({FBX_PATH!r})
# node = scene.nodes[{i}]
# print({i}, repr(str(node.name)))
# """
#     r = subprocess.run([sys.executable, "-c", node_code], capture_output=True, text=True, timeout=30)
#     if r.returncode != 0:
#         print(f"[CRASH] node {i}: code={r.returncode} stderr={r.stderr.strip()[-200:]}")
#     else:
#         print(r.stdout.strip())