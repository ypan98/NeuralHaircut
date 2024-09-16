import argparse
import open3d as o3d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--path_to_head', default='./implicit-hair-data/data/00050/final_head.ply', type=str) 
    parser.add_argument('--path_to_hair', default='./implicit-hair-data/data/00050/hair_outer.ply', type=str) 
    parser.add_argument('--target_num_vertices', default=10000, type=int)
    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    path_to_case = args.path_to_hair.split('hair_outer.ply')[0]

    # Simplify hair mesh
    hair_mesh = o3d.io.read_triangle_mesh(args.path_to_hair)
    hair_mesh_simplified = hair_mesh.simplify_quadric_decimation(args.target_num_vertices)
    o3d.io.write_triangle_mesh(path_to_case + 'hair_outer_remeshed.ply', hair_mesh_simplified)

    # Simplify head mesh if original head mesh has more than 10000 vertices
    head_mesh = o3d.io.read_triangle_mesh(args.path_to_head)
    if len(head_mesh.vertices) > 10000:
        head_mesh_simplified = head_mesh.simplify_quadric_decimation(args.target_num_vertices)
        o3d.io.write_triangle_mesh(path_to_case + 'final_head_remeshed.ply', head_mesh_simplified)
    else:
        o3d.io.write_triangle_mesh(path_to_case + 'final_head_remeshed.ply', head_mesh)