import os.path as osp

import torch

from configs import configs as cfg
from lib.models.smpl import SMPL
from lib.models.network import Network


def build_smpl(args, device, batch_size=None, gender='neutral'):
    if batch_size is None: batch_size = args.batch_size * args.seq_length
    body_model = SMPL(
        J_regressor=cfg.SMPL.REGRESSOR,
        model_path=cfg.PATHS.SMPL,
        gender=gender,
        batch_size=batch_size,
        create_transl=False).to(device)
    
    return body_model


def get_network(args):
    return Network(args).to(args.device)


def load_checkpoint(args, net, optimizer=None):
    assert osp.exists(args.checkpoint
        ), f"Pretrained checkpoint not found! {args.checkpoint} not exists."
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['model'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    print(f'Successfully loaded pretrained model ! {args.checkpoint}')
    return net, optimizer, start_epoch
