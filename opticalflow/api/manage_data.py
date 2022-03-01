import os.path as osp

import cv2

from opticalflow.core.dataset import KITTIDemoManager
from opticalflow.dataset import (KITTI, FlyingChairs, FlyingThings3D,
                                 MpiSintel, HD1K)


def load_data(args):
    return KITTIDemoManager.load_images(args.img_prefix, args)


def create_dataloader(data, args):
    return KITTIDemoManager.create_dataloader(data, args)


def output_data(imgs, output_dir):
    file_path = osp.join(output_dir, 'demo.jpg')
    cv2.imwrite(file_path, imgs)


def fetch_training_data(args, TRAIN_DS='C+T+K+S+H'):
    """Create the data loader for the corresponding trainign set."""

    if args.dataset == 'Chairs':
        aug_params = {
            'crop_size': args.image_size,
            'min_scale': -0.1,
            'max_scale': 1.0,
            'do_flip': True
        }
        training_data = FlyingChairs(
            aug_params, split='training', root=args.data_root)

    elif args.dataset == 'Things':
        aug_params = {
            'crop_size': args.image_size,
            'min_scale': -0.4,
            'max_scale': 0.8,
            'do_flip': True
        }
        clean_dataset = FlyingThings3D(
            aug_params, dstype='frames_cleanpass', root=args.data_root)
        final_dataset = FlyingThings3D(
            aug_params, dstype='frames_finalpass', root=args.data_root)
        training_data = clean_dataset + final_dataset

    elif args.dataset == 'Sintel':
        aug_params = {
            'crop_size': args.image_size,
            'min_scale': -0.2,
            'max_scale': 0.6,
            'do_flip': True
        }
        things = FlyingThings3D(
            aug_params, dstype='frames_cleanpass', root=args.data_root)
        sintel_clean = MpiSintel(
            aug_params,
            split='training',
            root=args.val_Sintel_root,
            dstype='clean')
        sintel_final = MpiSintel(
            aug_params, split='training', root=args.data_root, dstype='final')

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({
                'crop_size': args.image_size,
                'min_scale': -0.3,
                'max_scale': 0.5,
                'do_flip': True
            })
            hd1k = HD1K({
                'crop_size': args.image_size,
                'min_scale': -0.5,
                'max_scale': 0.2,
                'do_flip': True
            })
            train_dataset = 100 * sintel_clean + 100 * sintel_final + 200 * kitti + 5 * hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100 * sintel_clean + 100 * sintel_final + things

    elif args.dataset == 'KITTI':
        aug_params = {
            'crop_size': args.image_size,
            'min_scale': -0.2,
            'max_scale': 0.4,
            'do_flip': False
        }
        training_data = KITTI(
            aug_params, split='training', root=args.val_KITTI_root)

    elif args.dataset == 'K+H':

        kitti = KITTI(
            {
                'crop_size': args.image_size,
                'min_scale': -0.2,
                'max_scale': 0.4,
                'do_flip': False
            },
            root=args.val_KITTI_root)
        hd1k = HD1K(
            {
                'crop_size': args.image_size,
                'min_scale': -0.5,
                'max_scale': 0.2,
                'do_flip': True
            },
            root=args.val_HD1K_root)
        training_data = 20 * kitti + hd1k

    elif args.dataset == 'K+H+S':

        aug_params = {
            'crop_size': args.image_size,
            'min_scale': -0.2,
            'max_scale': 0.6,
            'do_flip': True
        }

        sintel_clean = MpiSintel(
            aug_params,
            split='training',
            root=args.val_Sintel_root,
            dstype='clean')
        kitti = KITTI(
            {
                'crop_size': args.image_size,
                'min_scale': -0.2,
                'max_scale': 0.4,
                'do_flip': False
            },
            root=args.val_KITTI_root)
        hd1k = HD1K(
            {
                'crop_size': args.image_size,
                'min_scale': -0.5,
                'max_scale': 0.2,
                'do_flip': True
            },
            root=args.val_HD1K_root)
        training_data = 42 * kitti + hd1k + sintel_clean


    print('Training with %d image pairs' % len(training_data))
    return training_data
