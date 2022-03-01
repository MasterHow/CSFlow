import torch.nn.functional as F
from collections import OrderedDict
import numpy as np


class InputPadder:
    """Pads images such that dimensions are divisible by 8 , from RAFT."""

    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [
                pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2,
                pad_ht - pad_ht // 2
            ]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def fill_order_keys(key, fill_value='_model.', fill_position=7):
    """fill order_dict keys in checkpoint, by Hao."""
    return key[0:fill_position] + fill_value + key[fill_position:]


def fix_order_keys(key, delete_value=6):
    """fix order_dict keys in checkpoint, by Hao."""
    return key[0:delete_value] + key[13:]


def fix_read_order_keys(key, start_value=7):
    """fix reading restored ckpt order_dict keys, by Hao."""
    return key[start_value:]
