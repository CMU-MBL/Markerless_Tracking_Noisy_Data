import os.path as osp

import torch
import numpy as np

from configs.train_options import TrainOptions
from lib.utils import data_utils as d_utils
from lib.data.totalcapture import get_totalcapture_dloader
from lib.core.criterion import Criterion
from lib.models.builder import (build_smpl, 
                                get_network, 
                                load_checkpoint)


@torch.no_grad()
def evaluate_on_totalcapture(net, args, epoch, winsize=0):
    criterion = Criterion(args)
    test_dloader = get_totalcapture_dloader(args)
    test_smpl = build_smpl(args, batch_size=(1 * args.seq_length), device=args.device)
    
    net.eval()
    mpjpe, mpjpe_pa, mpjae = [], [], []
    
    print('\n' + '#' * 10 + '  Evaluation  ' + '#' * 10)
    for batch in test_dloader:
        vid = batch['vid'][0]
        if 'walking' not in vid: continue

        total_len = batch['pose'].shape[1]
        inputs, groundtruths = d_utils.prepare_batch(args, batch)
        
        pred_kp3d, pred_pose = [], []
        for start, end in zip(range(0, total_len - args.seq_length, 2 * winsize + 1), 
                              range(args.seq_length, total_len, 2 * winsize + 1)):
            curr_inputs = dict()
            curr_groundtruths = dict()
            for key, val in inputs.items():
                curr_inputs[key] = val[:, start:end]
            for key, val in groundtruths.items():
                curr_groundtruths[key] = val[:, start:end]

            pred_smpl, _ = net(curr_inputs)
            output = test_smpl(pred_smpl, curr_groundtruths)
            
            idxs = [25 + i for i in range(-winsize, winsize + 1)]
            pred_kp3d.append(output.joints[idxs])
            pred_pose.append(pred_smpl[0][0, idxs])
            
        pred_kp3d = torch.cat(pred_kp3d).unsqueeze(0)
        pred_pose = torch.cat(pred_pose).unsqueeze(0)
        
        gt_kp3d = groundtruths['kp3d'][:, idxs[0]:][:, :pred_kp3d.shape[1]]
        gt_pose = groundtruths['pose'][:, idxs[0]:][:, :pred_kp3d.shape[1]]

        mpjpe.append(criterion._get_mpjpe(gt_kp3d, pred_kp3d, False, False))
        mpjpe_pa.append(criterion._get_pa_mpjpe(gt_kp3d, pred_kp3d, False))
        mpjae.append(criterion._get_mpjae(gt_pose, pred_pose)[0].item())
        
        print("{:<{}} | MPJPE : {:<{}}  MPJPE(PA) : {:<{}}  MPJAE : {:<{}}".format(
            vid, 15, f'{mpjpe[-1]:.2f}', 8, f'{mpjpe_pa[-1]:.2f}', 8, f'{mpjae[-1]:.2f}', 8
        ))
    
    print('')
    return {'epoch': epoch,
            'mpjpe': np.array(mpjpe).mean(), 
            'mpjpe_pa': np.array(mpjpe_pa).mean(),
            'mpjae': np.array(mpjae).mean()}


def main():
    args = TrainOptions().parse_args()
    net = get_network(args)
    net, _, epoch = load_checkpoint(args, net, None)
    
    evaluate_on_totalcapture(net, args, epoch, winsize=20)

if __name__ == '__main__':
    main()