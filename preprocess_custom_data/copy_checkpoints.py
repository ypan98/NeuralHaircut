from shutil import copyfile
import argparse
import os
from pathlib import Path

def main(args):

    case = args.case
    exps_dir = Path('./exp') / case / "first_stage"

    path_to_mesh = os.path.join(exps_dir, 'meshes')
    path_to_ckpt = os.path.join(exps_dir, 'checkpoints')
    path_to_fitted_camera = os.path.join(exps_dir, 'cameras')
    meshes = sorted(os.listdir(path_to_mesh))
    last_ckpt = sorted(os.listdir(path_to_ckpt))[-1]
    last_hair = [i for i in meshes if i.split('_')[-1].split('.')[0]=='hair'][-1]
    last_head = [i for i in meshes if i.split('_')[-1].split('.')[0]=='head'][-1]

    if not os.path.exists(f'./implicit-hair-data/data/{case}'):
        os.makedirs(f'./implicit-hair-data/data/{case}')
    copyfile(os.path.join(path_to_mesh, last_hair), f'./implicit-hair-data/data/{case}/final_hair.ply')
    copyfile(os.path.join(path_to_mesh, last_head), f'./implicit-hair-data/data/{case}/final_head.ply')
    copyfile(os.path.join(path_to_ckpt, last_ckpt), f'./implicit-hair-data/data/{case}/ckpt_final.pth')

    if os.path.exists(path_to_fitted_camera):
        print(f'Copy obtained from the first stage camera fitting checkpoint to folder ./implicit-hair-data/data/{case}')
        last_camera = sorted(os.listdir(path_to_fitted_camera))[-1]
        copyfile(os.path.join(path_to_fitted_camera, last_camera), f'./implicit-hair-data/data/{case}/fitted_cameras.pth')

    # copy head pior
    copyfile(os.path.join(args.path_to_scene, "head_prior.obj"), f'./implicit-hair-data/data/{case}/head_prior.obj')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--case', default='00050', type=str)
    parser.add_argument('--path_to_scene', default='../Datasets/usc_colmap/00050/neural_haircut', type=str)
    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)