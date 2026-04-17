import torch
import open3d as o3d

# load checkpoint
ckpt = torch.load("outputs/checkpoint.pt", map_location="cpu")

xyz = ckpt["xyz"]   # <-- now defined properly

pts = xyz.detach().cpu().numpy()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)

print("Total points:", len(pts))
print("Bounds:")
print("x:", pts[:,0].min(), pts[:,0].max())
print("y:", pts[:,1].min(), pts[:,1].max())
print("z:", pts[:,2].min(), pts[:,2].max())

o3d.visualization.draw_geometries([pcd])