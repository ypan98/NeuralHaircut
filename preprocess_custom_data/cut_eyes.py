import pytorch3d
import argparse
import os
from pytorch3d.io import load_obj, save_obj
import torch

def main(args):
    
    path_to_scene = os.path.join(args.path_to_scene)
    
    verts, faces, _ =  load_obj(os.path.join(path_to_scene, 'head_prior.obj'))
    idx_wo_eyes, faces_wo_eyes = torch.load('./data/idx_wo_eyes.pt'), torch.load('./data/faces_wo_eyes.pt')
    idx_wo_eyes = torch.tensor(idx_wo_eyes)

    save_obj(os.path.join(path_to_scene, 'head_prior_wo_eyes.obj'), verts[idx_wo_eyes], faces_wo_eyes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--path_to_scene', default='./implicit-hair-data/data/', type=str) 
    args, _ = parser.parse_known_args()
    args = parser.parse_args()
    main(args)