import open3d as o3d

pcd = o3d.io.read_point_cloud("outputs/donor_only.ply")
o3d.visualization.draw_geometries([pcd])