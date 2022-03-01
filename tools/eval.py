# Test and evaluate the model
# 1 load data
# 2.1 preprocess 2.2 init model 2.3 init evaluator
# 3 inference
# 4.1 postprocess 4.2 record loss
# 5.1 output image 5.2 evaluate

import argparse

from opticalflow.api import (manage_data, create_dataloader, inference,
                             init_evaluator, init_model, load_data,
                             output_data, postprocess_data, preprocess_data)
from opticalflow.api.evaluate import validate_chairs, validate_sintel, validate_kitti, validate_kitti12

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100


def parse_args():
    parser = argparse.ArgumentParser(description='Test and evaluate the model')
    parser.add_argument(
        '--model',
        help='The model use to inference',
        default='CSFlow')
    parser.add_argument(
        '--restore_ckpt',
        help="Restored checkpoint you are using/path or None",
        default='./checkpoints/CSFlow-things.pth')
    parser.add_argument(
        '--eval_iters',
        type=int,
        help='Iterations of GRU unit while eval, 32 for default',
        default=32)
    parser.add_argument(
        '--eval',
        default=True,
        help='Whether eval or test demo')
    parser.add_argument(
        '--validation',
        type=str,
        nargs='+',
        default=[],
        help='The dataset used to validate CSFlow',
        choices=['KITTI', 'KITTI12', 'Sintel', 'Chairs'])
    parser.add_argument(
        '--val_Sintel_root',
        help='Root of the current datasets')
    parser.add_argument(
        '--val_Chairs_root',
        help='Root of the current datasets')
    parser.add_argument(
        '--val_Things_root',
        help='Root of the current datasets')
    parser.add_argument(
        '--val_KITTI_root',
        help='Root of the current datasets')
    parser.add_argument(
        '--val_KITTI12_root',
        help='Root of the current datasets')
    parser.add_argument(
        '--train',
        help='True or False',
        default=True)
    parser.add_argument(
        '--dataset',
        help='The data use to train',
        default='Sintel',
        choices=['Chairs', 'Things', 'Sintel', 'KITTI'])
    parser.add_argument(
        '--change_gpu',
        help='train on cuda device but not cuda:0',
        action='store_true')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--DEVICE', help='The using device', default='cuda')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = init_model(args)

    results = {}
    for val_dataset in args.validation:
        if val_dataset == 'Chairs':
            if args.change_gpu:
                results.update(
                    validate_chairs(model, args.val_Chairs_root, args.gpus))
            else:
                results.update(
                    validate_chairs(model.module, args.val_Chairs_root))
        elif val_dataset == 'Sintel':
            if args.change_gpu:
                results.update(
                    validate_sintel(model, args.val_Sintel_root, args.gpus))
            else:
                results.update(
                    validate_sintel(model.module, args.val_Sintel_root))
        elif val_dataset == 'KITTI':
            if args.change_gpu:
                results.update(
                    validate_kitti(model, args.val_KITTI_root, args.gpus))
            else:
                results.update(
                    validate_kitti(model.module, args.val_KITTI_root))
        elif val_dataset == 'KITTI12':
            if args.change_gpu:
                results.update(
                    validate_kitti12(model, args.val_KITTI12_root, args.gpus))
            else:
                results.update(
                    validate_kitti12(model.module, args.val_KITTI12_root))


if __name__ == '__main__':
    main()
