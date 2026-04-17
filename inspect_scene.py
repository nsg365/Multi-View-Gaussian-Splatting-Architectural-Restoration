"""
inspect_scene.py — improved scene inspector (FIXED PICKING)
───────────────────────────────────────────────────────────
Now supports proper point picking + auto bounding box generation.

Controls:
  - Shift + Left Click : pick points
  - Q                  : finish selection
"""

import torch
import numpy as np
import open3d as o3d


# ── helper: point picking ─────────────────────────────────────────────────────
def pick_points(pcd, xyz, title="Pick points"):
    print("\n[INFO] Shift + Left Click to pick points")
    print("[INFO] Press 'Q' to finish\n")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title, width=1200, height=800)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    picked = vis.get_picked_points()

    print("\nPicked point indices:", picked)

    pts = []
    for idx in picked:
        p = xyz[idx]
        print(f"Point {idx}: x={p[0]:.3f}, y={p[1]:.3f}, z={p[2]:.3f}")
        pts.append(p)

    if len(pts) == 0:
        print("⚠️ No points selected.")
        return None

    pts = np.array(pts)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)

    print("\nComputed bounding box:")
    print(f"x: [{mins[0]:.2f}, {maxs[0]:.2f}]")
    print(f"y: [{mins[1]:.2f}, {maxs[1]:.2f}]")
    print(f"z: [{mins[2]:.2f}, {maxs[2]:.2f}]")

    return mins, maxs


# ── load checkpoint ────────────────────────────────────────────────────────────
ckpt    = torch.load("outputs/checkpoint.pt", map_location="cpu")
xyz_np  = ckpt["xyz"].numpy()

print(f"Total splats loaded: {len(xyz_np)}")
print(f"Raw scene bounds:")
print(f"  x: [{xyz_np[:,0].min():.2f}, {xyz_np[:,0].max():.2f}]")
print(f"  y: [{xyz_np[:,1].min():.2f}, {xyz_np[:,1].max():.2f}]")
print(f"  z: [{xyz_np[:,2].min():.2f}, {xyz_np[:,2].max():.2f}]")


# ── Step 1: outlier removal ───────────────────────────────────────────────────
pcd_raw = o3d.geometry.PointCloud()
pcd_raw.points = o3d.utility.Vector3dVector(xyz_np)

pcd_clean, _ = pcd_raw.remove_statistical_outlier(
    nb_neighbors=20, std_ratio=1.5
)
xyz_clean = np.asarray(pcd_clean.points)

print(f"\nAfter outlier removal: {len(xyz_clean)} splats")
print(f"Cleaned scene bounds:")
print(f"  x: [{xyz_clean[:,0].min():.2f}, {xyz_clean[:,0].max():.2f}]")
print(f"  y: [{xyz_clean[:,1].min():.2f}, {xyz_clean[:,1].max():.2f}]")
print(f"  z: [{xyz_clean[:,2].min():.2f}, {xyz_clean[:,2].max():.2f}]")


# ── Step 2: Z colouring ───────────────────────────────────────────────────────
z  = xyz_clean[:, 2]
zn = (z - z.min()) / (z.max() - z.min() + 1e-8)

colors = np.stack([1.0 - zn, zn, np.zeros_like(zn)], axis=1)
pcd_clean.colors = o3d.utility.Vector3dVector(colors)


# ── Step 3: bounding box + axes ───────────────────────────────────────────────
aabb = pcd_clean.get_axis_aligned_bounding_box()
aabb.color = (0.3, 0.3, 0.3)

centre = np.array(pcd_clean.get_center())
axis_len = np.ptp(xyz_clean, axis=0).max() * 0.3


def make_axis_line(start, end, color):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector([start, end])
    ls.lines  = o3d.utility.Vector2iVector([[0, 1]])
    ls.colors = o3d.utility.Vector3dVector([color])
    return ls


x_axis = make_axis_line(centre, centre + [axis_len, 0, 0], [1, 0, 0])
y_axis = make_axis_line(centre, centre + [0, axis_len, 0], [0, 1, 0])
z_axis = make_axis_line(centre, centre + [0, 0, axis_len], [0, 0, 1])


# ── Step 4: pick DAMAGED box ──────────────────────────────────────────────────
print("\n===== PICK DAMAGED STAIR =====")
damaged = pick_points(pcd_clean, xyz_clean, "Pick DAMAGED stair points")


# ── Step 5: pick DONOR box ────────────────────────────────────────────────────
print("\n===== PICK DONOR STAIR =====")
donor = pick_points(pcd_clean, xyz_clean, "Pick DONOR stair points")


# ── Step 6: print final usable boxes ──────────────────────────────────────────
if damaged and donor:
    dmin, dmax = damaged
    omin, omax = donor

    print("""
════════════════════════════════════════════════════════════
FINAL BOXES (copy this into reconstruct_stairs.py)
════════════════════════════════════════════════════════════
""")

    print(f"DAMAGED_BOX = ({dmin[0]:.2f}, {dmax[0]:.2f}, "
          f"{dmin[1]:.2f}, {dmax[1]:.2f}, "
          f"{dmin[2]:.2f}, {dmax[2]:.2f})")

    print(f"DONOR_BOX = ({omin[0]:.2f}, {omax[0]:.2f}, "
          f"{omin[1]:.2f}, {omax[1]:.2f}, "
          f"{omin[2]:.2f}, {omax[2]:.2f})")

    print("""
MIRROR_AXIS = "y"   # change if needed
OFFSET      = (0.0, 0.0, 0.0)
""")