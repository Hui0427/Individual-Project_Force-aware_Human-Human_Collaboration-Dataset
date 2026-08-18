# scripts/parse_motive_props.py
"""Parse Motive .motive rigid-body profiles into a single JSON asset.

Extracts, per rigid body:
  - marker offsets relative to the rigid-body pivot (metres, Motive right-handed Y-up)
  - the manual mesh-alignment fields (GeometryFile / Scale / Offset / PitchYawRoll)

Motive stores GeometryScale as a percentage (100 = unscaled); this is confirmed by
the YCB objects (known to be modelled in metres) all carrying 100,100,100.
"""
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

OUT_PATH = Path("outputs/motive_props.json")

# Sources are merged in order, so later ones win on name collisions. The alphabetical
# export is the current authoritative profile (36 rigid bodies in one file); the older
# per-object folder is kept because a few takes were recorded against those definitions
# and their Geometry fields are still the only alignment some objects have.
SOURCES = [
    Path("/Users/hxy/Desktop/YCB/motive prop/algined"),
    Path("/Users/hxy/Desktop/YCB/Assets_alphabetical.motive"),
]


def _floats(text):
    return [float(v) for v in re.split(r"[,\s]+", text.strip()) if v]


def parse_rigid_body(rb):
    props = {p.get("name"): p.get("value") for p in rb.findall("./properties/property")}

    markers = []
    for c in rb.findall("./constraints/constraint"):
        markers.append(
            {
                "name": c.get("name"),
                "offset_m": _floats(c.findtext("offset")),
                "diameter_m": float(c.findtext("diameter")),
            }
        )

    geom = None
    if "GeometryFile" in props:
        scale_pct = _floats(props.get("GeometryScale", "100,100,100"))
        pyr = _floats(props["GeometryPitchYawRoll"]) if "GeometryPitchYawRoll" in props else [0.0, 0.0, 0.0]
        geom = {
            "file_windows_path": props["GeometryFile"],
            "file_basename": props["GeometryFile"].replace("\\", "/").rsplit("/", 1)[-1],
            "scale_percent": scale_pct,
            "scale_factor": [s / 100.0 for s in scale_pct],
            "offset_m": _floats(props["GeometryOffset"]) if "GeometryOffset" in props else [0.0, 0.0, 0.0],
            "pitch_yaw_roll_deg": pyr,
            "pitch_yaw_roll_explicit": "GeometryPitchYawRoll" in props,
            # GeometryOffset is the field that cannot be defaulted, so its presence is
            # what marks the manual alignment as done. A missing PitchYawRoll is a
            # legitimate zero (football: sphere, verified numerically at 6.4 mm).
            "alignment_done": "GeometryOffset" in props,
        }

    return {
        "asset_name": props.get("AssetName") or props.get("NodeName"),
        "num_markers": len(markers),
        "markers": markers,
        "geometry": geom,
    }


def key_for(entry, path):
    """Objects are keyed by asset name, lowercased with spaces -> underscores, so a
    multi-body export and the one-body-per-file folder land in the same namespace."""
    name = entry["asset_name"] or path.stem
    return name.strip().lower().replace(" ", "_")


def main():
    sources = [Path(p) for p in sys.argv[1:]] or SOURCES
    files = []
    for s in sources:
        files.extend(sorted(s.glob("*.motive")) if s.is_dir() else [s])

    # The same object can exist as several rigid-body *definitions* (rebuilt in Motive
    # between sessions, sometimes with a different marker count and mesh). Never let an
    # unaligned definition displace an aligned one - which definition a given take
    # actually used is decided later, by fitting its .skel markers.
    out = {"sources": [str(s) for s in sources], "objects": {}, "variants": {}}
    for path in files:
        root = ET.parse(path).getroot()
        for rb in root.findall(".//rigid_body"):
            entry = parse_rigid_body(rb)
            entry["source_file"] = path.name
            key = key_for(entry, path)
            prev = out["objects"].get(key)
            if prev is None:
                out["objects"][key] = entry
                continue
            new_ok = bool(entry["geometry"] and entry["geometry"]["alignment_done"])
            old_ok = bool(prev["geometry"] and prev["geometry"]["alignment_done"])
            keep, drop = (entry, prev) if (new_ok or not old_ok) else (prev, entry)
            out["objects"][key] = keep
            out["variants"].setdefault(key, []).append(
                {"source_file": drop["source_file"], "num_markers": drop["num_markers"],
                 "aligned": bool(drop["geometry"] and drop["geometry"]["alignment_done"]),
                 "markers": drop["markers"],
                 "geometry": drop["geometry"]})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))

    aligned = [n for n, o in out["objects"].items()
               if o["geometry"] and o["geometry"]["alignment_done"]]
    for name in sorted(out["objects"]):
        o = out["objects"][name]
        g = o["geometry"]
        if g is None:
            tag = "no geometry"
        else:
            tag = (f"{g['file_basename']}  scale={g['scale_percent']}  "
                   f"pyr={'set' if g['pitch_yaw_roll_explicit'] else 'ABSENT'}"
                   f"{'' if g['alignment_done'] else '   <-- 未对齐,不可用'}")
        print(f"{name:24s} {o['num_markers']} markers  {tag}   ({o['source_file']})")
    print(f"\n{len(aligned)}/{len(out['objects'])} 个物体有可用的对齐: {sorted(aligned)}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
