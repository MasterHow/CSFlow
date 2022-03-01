from collections import OrderedDict

import torch

from opticalflow.core.model import csflow
from opticalflow.utils.utils import fill_order_keys, fix_read_order_keys


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_CSFlow(args):
    if args.train:
        if args.change_gpu:
            device = torch.device(('cuda:' + str(args.gpus[0])))
            model = raft.RAFT(args)
            model.to(device)
            print('Parameter Count: %d' % count_parameters(model))

            # read checkpoint
            if args.restore_ckpt is not None:
                try:
                    model.load_state_dict(torch.load(args.restore_ckpt))
                except (Exception):
                    try:
                        # 修改OrderDict索引来读入checkpoint
                        d1 = torch.load(args.restore_ckpt)
                        d2 = OrderedDict([
                            (fill_order_keys(k, fill_value='_model.'), v)
                            for k, v in d1.items()
                        ])
                        model.load_state_dict(d2)
                    except:
                        try:
                            # 修改OrderDict索引来读入checkpoint
                            d1 = torch.load(args.restore_ckpt)
                            d2 = OrderedDict([
                                (fix_read_order_keys(k, start_value=7), v)
                                for k, v in d1.items()
                            ])
                            model.load_state_dict(d2)
                        except:
                            # 修改OrderDict索引来读入checkpoint
                            d1 = torch.load(args.restore_ckpt)
                            d2 = OrderedDict([(fill_order_keys(
                                fix_read_order_keys(k, start_value=7),
                                fill_value='_model.',
                                fill_position=0), v) for k, v in d1.items()])
                            model.load_state_dict(d2)

                pass

            model.to(device)
            model.train()

            if args.dataset != 'Chairs':
                # model.module.freeze_bn()
                # 调用错误, 等价实现冻结BN层
                for m in model.modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        m.eval()

            return model
        else:
            model = torch.nn.DataParallel(
                csflow.CSFlow(args), device_ids=args.gpus)
            print('Parameter Count: %d' % count_parameters(model))

            # read checkpoint
            if args.restore_ckpt is not None:
                try:
                    model.load_state_dict(torch.load(args.restore_ckpt))
                except (Exception):
                    # 修改OrderDict索引来读入checkpoint
                    try:
                        d1 = torch.load(args.restore_ckpt)
                        d2 = OrderedDict([
                            (fill_order_keys(k, fill_value='_model.'), v)
                            for k, v in d1.items()
                        ])
                        model.load_state_dict(d2)
                    except (Exception):
                        d1 = torch.load(args.restore_ckpt, map_location='cpu')
                        d2 = OrderedDict([(fill_order_keys(
                            k, fill_value='module.', fill_position=0), v)
                                          for k, v in d1.items()])
                        model.load_state_dict(d2)
                pass

            model.cuda()
            model.train()

            if args.dataset != 'Chairs':
                # model.module.freeze_bn()
                # 调用错误, 等价实现冻结BN层
                for m in model.modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        m.eval()

            return model
    else:
        return csflow.CSFlow(args)


def init_model(args):
    return init_CSFlow(args)
