import os
import os.path as osp
from collections import defaultdict


import torch
import numpy as np
from tqdm import tqdm

from configs import configs as cfg
from lib.utils import transforms


def main():
    _, dataset_list, _ = next(os.walk(cfg.PATHS.AMASS_BASE))
    dataset_list.remove('TotalCapture')     # TotalCapture is validation/test set
    
    vid_idx = 0
    outfile = defaultdict(list)
    
    for dataset in (dataset_pbar := tqdm(sorted(dataset_list))):
        dataset_pbar.set_description(f'Dataset - {dataset}')
        
        dataset_fldr = osp.join(cfg.PATHS.AMASS_BASE, dataset)
        _, subj_list, _ = next(os.walk(dataset_fldr))
        subj_list.sort()
        
        for subj in (subj_pbar := tqdm(sorted(subj_list), leave=False)):
            subj_pbar.set_description(f'Subject - {subj}')
            
            action_list = [x for x in os.listdir(osp.join(dataset_fldr, subj)) if x.endswith('.npz')]
            action_list.sort()
            
            for action in (action_pbar := tqdm(sorted(action_list), leave=False)):
                action_pbar.set_description(f'Action - {action}')
                
                label_fname = osp.join(dataset_fldr, subj, action)
                if label_fname.endswith('shape.npz') or label_fname.endswith('stagei.npz'): 
                    continue
                
                label = np.load(label_fname, allow_pickle=True)
                try: mocap_framerate = int(label['mocap_frame_rate'])
                except: mocap_framerate = int(label['mocap_framerate'])
                retain_freq = max(1, mocap_framerate // cfg.CONSTANTS.FPS)
                
                # Pose
                n_frames = label['poses'].shape[0]
                pose = torch.from_numpy(label['poses']).float().reshape(n_frames, -1, 3)[:, amass_joint_idxs, :]
                pose = transforms.axis_angle_to_matrix(pose)[::retain_freq]
                n_frames = pose.shape[0]
                
                # Shape
                betas = torch.from_numpy(label['betas'][:10]).unsqueeze(0).float().expand(n_frames, -1)
                
                # Transl
                transl = torch.from_numpy(label['trans']).float()[::retain_freq]
                
                if pose.shape[0] < 100:
                    continue
                
                outfile['pose'].append(pose)
                outfile['betas'].append(betas)
                outfile['transl'].append(transl)
                outfile['vid'].append(torch.tensor([vid_idx] * pose.shape[0]).to(dtype=torch.int16))
                
    for key in outfile.keys():
        outfile[key] = torch.cat(outfile[key], dim=0)
        
    torch.save(outfile, cfg.PATHS.AMASS_LABEL)
                


if __name__ == '__main__':
    amass_joint_idxs = list(range(24)); amass_joint_idxs[-1] = 37
    main()