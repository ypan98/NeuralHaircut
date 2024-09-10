import trimesh
import numpy as np
import pickle
import argparse
import os


def main(args):
    filename = 'point_cloud_cropped.ply' if args.cropped else 'point_cloud.ply'
    pc = np.array(trimesh.load(os.path.join(args.path_to_scene, filename)).vertices)

    translation = (pc.min(0) + pc.max(0)) / 2
    scale = np.linalg.norm(pc - translation, axis=-1).max().item() / 1.1

    tr = (pc - translation) / scale
    assert tr.min() >= -1 and tr.max() <= 1

    print('Scaling into the sphere', tr.min(), tr.max())

    d = {'scale': scale,
        'translation': list(translation)}

    with open(os.path.join(args.path_to_scene, 'scale.pickle'), 'wb') as f:
        pickle.dump(d, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--path_to_scene', default='../Datasets/usc_colmap/00050', type=str) 
    parser.add_argument('--cropped', action='store_true', default=False)
    
    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)