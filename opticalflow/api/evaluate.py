import os

import numpy as np
import torch
from torch.nn.functional import interpolate

import opticalflow.dataset as dataset
from opticalflow.utils.flow_utils import (forward_interpolate, writeFlow,
                                          writeFlowKITTI)
from opticalflow.utils.utils import InputPadder

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


class Evaluator():

    def __init__(self, data_size: int = None):
        self._loss_list = []
        self._data_size = data_size
        self._count = 0
        self._data_completed_count = 0

    def record_result(self, y, y_gt):
        pass

    def record_loss(self, loss, current_batch_size):
        self._loss_list.append(loss)
        self._data_completed_count += current_batch_size

    def print_current_evaluation_result(self):
        self._count += 1
        loss = self._loss_list[-1]
        print(f'({self._count}) loss: {loss:>7f}  '
              f'[{self._data_completed_count:>5d}/{self._data_size:>5d}]')

    def print_all_evaluation_result(self):
        print('Looks good')


def init_evaluator(data_size=1):
    return Evaluator(data_size)


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """Loss function defined over sequence of flow predictions, from RAFT."""

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


@torch.no_grad()
def create_sintel_submission(model,
                             warm_start=False,
                             output_path='sintel_submission'):
    """Create submission for the Sintel leaderboard, from RAFT, modified."""
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = dataset.MpiSintel(
            split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(),
                                        image2[None].cuda())

            # zip image
            image_pair = torch.stack((image1, image2))

            flow_low, flow_pr = model._model(
                image_pair, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir,
                                       'frame%04d.flo' % (frame + 1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, data_root, output_path='kitti_submission'):
    """Create submission for the kitti leaderboard, from RAFT, modified."""
    model.eval()
    test_dataset = dataset.KITTI(
        split='testing', aug_params=None, root=data_root)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        # zip image
        image_pair = torch.stack((image1, image2))

        _, flow_pr = model._model(image_pair, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, data_root, gpus=[0]):
    """Perform evaluation on the FlyingChairs (test) split, from RAFT,
    modified."""
    model.eval()
    epe_list = []

    val_dataset = dataset.FlyingChairs(split='validation', root=data_root)
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda(gpus[0])
        image2 = image2[None].cuda(gpus[0])

        # zip image
        image_pair = torch.stack((image1, image2))

        _, flow_pr = model._model(image_pair, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print('Validation Chairs EPE: %f' % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, data_root, gpus=[0]):
    """Peform validation using the Sintel (train) split, from RAFT, ,
    modified."""
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = dataset.MpiSintel(
            split='training', root=data_root, dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda(gpus[0])
            image2 = image2[None].cuda(gpus[0])

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # zip image
            image_pair = torch.stack((image1, image2))

            flow_low, flow_pr = model._model(image_pair, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        print('Validation Sintel (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f' %
              (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_kitti(model, data_root, gpus=[0]):
    """Peform validation using the KITTI-2015 (train) split, from RAFT,
    modified."""
    model.eval()
    val_dataset = dataset.KITTI(split='training', root=data_root)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda(gpus[0])
        image2 = image2[None].cuda(gpus[0])

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        # zip image
        image_pair = torch.stack((image1, image2))

        flow_low, flow_pr = model._model(image_pair, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print('Validation KITTI: %f, %f' % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


@torch.no_grad()
def validate_kitti12(model, data_root, gpus=[0]):
    """Peform validation using the KITTI-2012 (train) split, by Hao."""
    model.eval()
    val_dataset = dataset.KITTI12(split='training', root=data_root)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda(gpus[0])
        image2 = image2[None].cuda(gpus[0])

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        # zip image
        image_pair = torch.stack((image1, image2))

        flow_low, flow_pr = model._model(image_pair, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print('Validation KITTI-12: %f, %f' % (epe, f1))
    return {'kitti-12-epe': epe, 'kitti-12-f1': f1}

