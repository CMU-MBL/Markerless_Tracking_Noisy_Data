import os
import os.path as osp
import logging
from collections import defaultdict

from glob import glob

import torch
import numpy as np
from tqdm import tqdm

from configs import configs as cfg
from lib.utils import transforms
from lib.models.builder import build_smpl
from lib.data.data_utils.gendb import (process_imu_data,
                                       process_video_data
)

test_actions = {
    's1': ['acting1', 'freestyle2', 'rom3', 'walking1'],
    's2': ['acting2', 'freestyle2', 'rom2', 'walking2'],
    's3': ['acting2', 'rom3', 'walking3'],
    's4': ['freestyle3', 'rom3', 'walking2'],
    's5': ['freestyle3', 'rom3', 'walking2']
}


logger = None
def write_log(rmse_acc, rmse_gyr, rmse_kp3d, subj, action):
    global logger
    if logger is None:
        # Create Logger
        os.makedirs('output/data_processing', exist_ok=True)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler('output/data_processing/totalcapture.log', mode='w')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    msg = "{:<{}}".format(f'{subj}_{action}', 15)
    msg += "|  "
    msg += "{:<{}}".format(f'Acc: {rmse_acc:.2f}', 15)
    msg += "{:<{}}".format(f'Gyr: {rmse_gyr:.2f}', 15)
    msg += "{:<{}}".format(f'Kp: {rmse_kp3d:.2f}', 15)
    
    logger.debug(msg)


def main(dtype='train'):
    retain_n_frames = 60 // cfg.CONSTANTS.FPS
    
    vid_idx = 0
    outfile = defaultdict(list)
    _, subj_list, _ = next(os.walk(osp.join(cfg.PATHS.AMASS_BASE, 'TotalCapture')))
    
    for subj in (subj_pbar := tqdm(sorted(subj_list))):
        subj_pbar.set_description(f'Subject - {subj.upper()}')
        label_fname_list = glob(osp.join(cfg.PATHS.AMASS_BASE, 'TotalCapture', subj, '*.npz'))
        
        for label_fname in (action_pbar := tqdm(sorted(label_fname_list), leave=False)):
            action = label_fname.split('/')[-1].split('_')[0]
            action_pbar.set_description(f'Action  - {action}')
            if (dtype == 'test') ^ (action in test_actions[subj]): continue
            
            # Load AMASS label
            label = np.load(label_fname, allow_pickle=True)
            n_frames = label['poses'].shape[0]
            
            # Pose
            pose = torch.from_numpy(label['poses']).float().reshape(n_frames, -1, 3)[:, amass_joint_idxs, :]
            pose = transforms.axis_angle_to_matrix(pose)
            
            # Shape
            betas = torch.from_numpy(label['betas'][:10]).unsqueeze(0).float().expand(n_frames, -1)
            
            # Transl
            transl = torch.from_numpy(label['trans']).float()
            
            # Some sequences are not synchronized
            if subj == 's5' and action == 'freestyle3':
                pose = pose[43:]; betas = betas[43:]; transl = transl[43:]
                
            # Get SMPL output
            smpl = build_smpl(None, 'cpu', n_frames)
            output = smpl.get_output(global_orient=pose[:, :1], body_pose=pose[:, 1:], betas=betas, transl=transl, pose2rot=False)
                
            # Process IMU data
            ori, acc, gyr, acc_v, gyr_v, sync_imu, rmse_acc, rmse_gyr = process_imu_data(
                smpl, output, pose, subj, action, viz=viz)
            
            # Using the syncing point, shift AMASS data
            pose = torch.roll(pose, shifts=sync_imu, dims=0)
            transl = torch.roll(transl, shifts=sync_imu, dims=0)
            output = smpl.get_output(global_orient=pose[:, :1], body_pose=pose[:, 1:], betas=betas, transl=transl, pose2rot=False)
            
            # Process Video data
            gt_kp3d = output.joints
            kp3d, rmse_kp3d = process_video_data(gt_kp3d, subj, action, viz=viz)
            
            l = min(pose.shape[0], acc.shape[0], kp3d.shape[0])
            for key in ('pose', 'betas', 'transl', 'gt_kp3d', 'kp3d', 'ori', 'acc', 'gyr', 'acc_v', 'gyr_v'):
                data = locals()[key][abs(sync_imu):l-abs(sync_imu)]
                outfile[key].append(data[::retain_n_frames])
            
            if dtype == 'train':
                outfile['vid'].append(torch.tensor([vid_idx] * outfile['pose'][-1].shape[0]).to(dtype=torch.int16))
                vid_idx += 1
            else:
                outfile['vid'].append(f'{subj}_{action}')
            
            # Write log
            write_log(rmse_acc, rmse_gyr, rmse_kp3d, subj, action)
    
    torch.save(outfile, cfg.PATHS.TC_LABEL[dtype])


if __name__ == '__main__':
    viz = False
    amass_joint_idxs = list(range(24)); amass_joint_idxs[-1] = 37
    main('test')