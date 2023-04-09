import random
import os.path as osp

import torch
import numpy as np

from configs.train_options import TrainOptions
from lib.utils.checkpoint import Logger
from lib.utils import data_utils as d_utils
from lib.data.amass import get_amass_dloader
from lib.core.criterion import Criterion
from lib.core.evaluate import evaluate_on_totalcapture
from lib.models.builder import (build_smpl, 
                                get_network, 
                                load_checkpoint)

def setup_seed(seed):
    """ Setup seed for reproducibility """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_one_epoch(args, net, train_dloader, optimizer, criterion, logger, epoch):
    net.train()

    for _iter, batch in enumerate(train_dloader):
        if batch['pose'].shape[0] != args.batch_size: continue

        inputs, groundtruths = d_utils.prepare_batch(args, batch)
        pred_smpl, unimodals = net(inputs)

        output = train_smpl(pred_smpl, groundtruths)
        pred_kp3d = output.joints.view(args.batch_size, args.seq_length, -1, 3)
        
        total_loss, loss_dict = criterion(pred_smpl, pred_kp3d, groundtruths)
        if args.model_type != 'imu':
            loss_dict['Input Keypoints Qty'] = criterion._get_mpjpe(
                inputs['kp3d'], groundtruths['kp3d'])

        if args.model_type == 'fusion' and epoch < args.supervise_unimodal_until_n_epoch:
            loss_v, loss_dict_v = criterion((unimodals[0], None, None), None, groundtruths)
            loss_i, loss_dict_i = criterion((unimodals[1], None, None), None, groundtruths)
            total_loss += (loss_v + loss_i)

            for key, val in loss_dict_v.items():
                loss_dict[f'{key} (Video)'] = val
            for key, val in loss_dict_i.items():
                loss_dict[f'{key} (IMU)'] = val

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        logger(loss_dict, _iter, epoch)


def main(start_epoch):
    net = get_network(args)
    
    train_dloader = get_amass_dloader(args)
    
    criterion = Criterion(args)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.lr, betas=(args.betas, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.5)

    if osp.exists(args.checkpoint):
        net, optimizer, start_epoch = load_checkpoint(args, net, optimizer)
    
    logger = Logger(model_dir=osp.join(args.logdir, args.name),
                    write_freq=args.write_freq,
                    checkpoint_freq=args.checkpoint_freq,
                    total_iteration=len(train_dloader))
    logger._save_configs(args)

    # This model doesn't predict wrist joint. Set as default value 0.
    global train_smpl
    train_smpl = build_smpl(args, args.device)

    for epoch in range(start_epoch, args.epoch + 1):
        train_one_epoch(args, net, train_dloader, optimizer, criterion, logger, epoch)
        val_results = evaluate_on_totalcapture(net, args, epoch, winsize=20)
        logger._save_checkpoint(net, {'optimizer': optimizer},
                                eval_dict=val_results, epoch=epoch)
        train_dloader.dataset.prepare_video_batch()
        
        if epoch % 5 == 0:
            scheduler.step()
        
        
if __name__ == '__main__':
    setup_seed(42)
    args = TrainOptions().parse_args()
    main(start_epoch=1)