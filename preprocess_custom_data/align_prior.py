import numpy as np
import torch
import open3d as o3d
import argparse
from pytorch3d.ops import iterative_closest_point


if __name__ == '__main__':
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument("--head_prior_path", default="implicit-hair-data/data/293/head_prior_aligned.obj", type=str)
    parser.add_argument("--head_mesh_path", default="implicit-hair-data/data/293/final_head_remeshed.ply", type=str)
    parser.add_argument("--vis3d", action="store_true")
    args = parser.parse_args()

    head_prior_o3d = o3d.io.read_triangle_mesh(args.head_prior_path)
    head_mesh_o3d = o3d.io.read_triangle_mesh(args.head_mesh_path)
    head_prior_verts = np.asarray(head_prior_o3d.vertices)
    head_mesh_verts = np.asarray(head_mesh_o3d.vertices)

    # find transformation with ICP
    X = torch.from_numpy(head_prior_verts).unsqueeze(0).to('cuda')
    Y = torch.from_numpy(head_mesh_verts).unsqueeze(0).to('cuda')
    solution = iterative_closest_point(X, Y, max_iterations=10000, estimate_scale=False)
    if not solution.converged:
        print(f"ICP did not converge, rmse {solution.rmse.item()}")
    R = solution.RTs.R[0].cpu().numpy()
    T = solution.RTs.T[0].cpu().numpy()
    S = solution.RTs.s[0].cpu().numpy()
    print(R, T, S)

    # new_head_prior_verts = (S * (head_prior_verts @ R)) + T
    # head_prior_o3d.vertices = o3d.utility.Vector3dVector(new_head_prior_verts)
    if args.vis3d:
        # visualize the aligned data with red and head mesh as blue
        head_prior_o3d.paint_uniform_color([1, 0, 0])
        head_mesh_o3d.paint_uniform_color([0, 0, 1])
        o3d.visualization.draw_geometries([head_prior_o3d, head_mesh_o3d])

    # # save the aligned data to ply
    # output_path = args.nh_data_path.replace(".ply", "_aligned.ply")
    # print(f"Saving aligned data to {output_path}")
    # points_poly.save(output_path, recompute_normals=False)