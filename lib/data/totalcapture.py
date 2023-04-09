import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from configs import configs as cfg
from lib.utils.data_utils import (get_smpl,
                                  make_collate_fn,
)


class TotalCapture(Dataset):
    def __init__(self, args, val=False):
        self.args = args
        self.val = val
        self.model_type = args.model_type
        self.joint_type = args.joint_type
        self.labels = torch.load(cfg.PATHS.TC_LABEL['val' if val else 'test'])
        get_smpl(args)
            
    def __len__(self):
        return len(self.labels['pose'])
        
    def __getitem__(self, index): 
        # Groundtruths
        pose = self.labels['pose'][index]
        transl = self.labels['transl'][index]
        betas = self.labels['betas'][index]
        gt_kp3d = self.labels['gt_kp3d'][index]
        
        # Video data
        kp3d = self.labels['kp3d'][index]
        
        # IMU data
        acc = self.labels['acc'][index]
        gyr = self.labels['gyr'][index]
        ori = self.labels['ori'][index]
               
        return {
            'pose': pose[1:-1, cfg.SMPL.MAIN_JOINTS],
            'transl': transl[1:-1],
            'betas': betas[1:-1],
            'acc': acc[1:-1] / 9.81,
            'gyr': gyr[1:-1] / 10,
            'ori': ori[1:-1],
            'kp3d': kp3d[1:-1],
            'gt_kp3d': gt_kp3d[1:-1],
            'vid': self.labels['vid'][index]
        }

        
        
def get_totalcapture_dloader(args, val=False):
    print('Load Total Capture Dataset...')
    
    dloader = DataLoader(TotalCapture(args, val),
                         batch_size=args.batch_size if val else 1,
                         shuffle=val,
                         num_workers=args.num_workers,
                         pin_memory=args.pin_memory,
                         collate_fn=make_collate_fn())
    
    return dloader