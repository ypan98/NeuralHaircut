import warnings
import os
import torch
from dataset import Multiview_dataset
from losses import OpenPoseLoss, RegShapeLoss, KeypointsMatchingLoss
from runner import Runner
import argparse
from pyhocon import ConfigFactory
from models.SMPLX import SMPLX
from utils.config import cfg as pixie_cfg
warnings.filterwarnings("ignore")


def main(conf, batch_size, train_rotation, train_pose, train_shape, checkpoint_path, save_path):
    # with open(conf_path) as f:
    #     conf_text = f.read()
    # conf = ConfigFactory.parse_string(conf_text)
    device = torch.device(conf['general']['device'])

    # Create SMPLX model from PIXIE
    smplx_model = SMPLX(pixie_cfg.model).to(device) 

    dataset = Multiview_dataset(**conf['dataset'], device=conf['general']['device'], batch_size=batch_size)

    # Create train losses
    losses = []
    if conf['loss']['fa_kpts_2d_weight']:
        losses += [KeypointsMatchingLoss(device=device, use_3d=False)]
    if conf['loss']['fa_kpts_3d_weight']:
        losses += [KeypointsMatchingLoss(device=device, use_3d=True)]
    # if conf.get_float('loss.openpose_face_weight'):
    #     losses += [OpenPoseLoss(mode='face', device=device)]
    # if conf.get_float('loss.openpose_body_weight'):
    #     losses += [OpenPoseLoss(mode='body', device=device)]
    if conf['loss']['reg_shape_weight']:
        losses += [RegShapeLoss(weight=conf['loss']['reg_shape_weight'])]
    loss_weights = {}
    loss_weights['reg_shape'] = conf['loss']['reg_shape_weight']
    # loss_weights['openpose_body'] = conf.get_float('loss.openpose_body_weight')
    # loss_weights['openpose_face'] = conf.get_float('loss.openpose_face_weight')
    loss_weights['fa_kpts'] = conf['loss']['fa_kpts_2d_weight']

    os.makedirs(save_path, exist_ok=True)

    runner = Runner(
            dataset,
            losses,
            smplx_model,
            device,
            save_path,
            cut_flame_head=conf['general']['cut_flame_head'],
            loss_weights=loss_weights,
            train_rotation=train_rotation, 
            train_shape=train_shape, 
            train_pose=train_pose, 
            checkpoint_path=checkpoint_path
        )
    runner.fit(
            epochs=conf['train']['epochs'],
            lr=conf['train']['learning_rate'],
            max_iter=conf['train']['max_iter'],
        )
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_rotation', type=bool, default=True)
    parser.add_argument('--train_pose', type=bool, default=False)
    parser.add_argument('--train_shape', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')

    # General
    parser.add_argument('--device', type=str, default='cuda', help='Device to use, e.g., cpu or cuda')
    parser.add_argument('--cut_flame_head', type=bool, default=True, help='Whether to cut flame head or not')
    # Dataset paths
    parser.add_argument('--data_path', type=str, default='../Datasets/usc_colmap/00050/neural_haircut', help='Path to the image dataset')
    # Optional arguments (commented in the original)
    parser.add_argument('--openpose_kp_path', type=str, default=None, help='Path to openpose keypoints file (optional)')
    parser.add_argument('--fitted_camera_path', type=str, default=None, help='Path to fitted camera parameters (optional)')
    parser.add_argument('--views_idx', type=str, default=None, help='Views index (optional)')
    # Loss weights
    parser.add_argument('--fa_kpts_2d_weight', type=float, default=1.0, help='Weight for 2D face keypoints loss')
    parser.add_argument('--fa_kpts_3d_weight', type=float, default=0.0, help='Weight for 3D face keypoints loss')
    parser.add_argument('--openpose_face_weight', type=float, default=1.0, help='Weight for openpose face keypoints')
    parser.add_argument('--openpose_body_weight', type=float, default=0.0, help='Weight for openpose body keypoints')
    parser.add_argument('--reg_shape_weight', type=float, default=0.000, help='Regularization weight for shape')
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--max_iter', type=int, default=500, help='Maximum number of iterations')
    parser.add_argument('--learning_rate', type=float, default=0.5, help='Learning rate for training')

    args = parser.parse_args()

    config = {
        'general': {
            'device': args.device,
            'cut_flame_head': args.cut_flame_head
        },
        'dataset': {
            'image_path': args.data_path + "/full_res_image",
            'scale_path': args.data_path + "/scale.pickle",
            'camera_path': args.data_path + "/cameras.npz",
            'pixie_init_path': args.data_path + "/initialization_pixie",
            'openpose_kp_path': args.openpose_kp_path,
            'fitted_camera_path': args.fitted_camera_path,
            'views_idx': args.views_idx
        },
        'loss': {
            'fa_kpts_2d_weight': args.fa_kpts_2d_weight,
            'fa_kpts_3d_weight': args.fa_kpts_3d_weight,
            'openpose_face_weight': args.openpose_face_weight,
            'openpose_body_weight': args.openpose_body_weight,
            'reg_shape_weight': args.reg_shape_weight
        },
        'train': {
            'epochs': args.epochs,
            'max_iter': args.max_iter,
            'learning_rate': args.learning_rate
        }
    }

    
    main(
        config,
        batch_size=args.batch_size,
        train_rotation=args.train_rotation, 
        train_pose=args.train_pose,
        train_shape=args.train_shape,
        checkpoint_path=args.checkpoint_path,
        save_path = args.save_path
     )