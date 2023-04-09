import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage.util.shape import view_as_windows

from configs import configs as cfg
from lib.utils.data_utils import (get_smpl,
                                  make_collate_fn,
                                  augment_IMU,
                                  augment_keypoints
)


class TotalCapture(Dataset):
    def __init__(self, args, train=False):
        self.args = args
        self.train = train
        self.model_type = args.model_type
        self.joint_type = args.joint_type
        self.labels = torch.load(cfg.PATHS.TC_LABEL['train' if train else 'test'])
        get_smpl(args)
        
        if self.train:
            self.prepare_video_batch()
    
    def __len__(self):
        if self.train:
            return len(self.video_indices)
        else:
            return len(self.labels['pose'])
    
    def __getitem__(self, index):
        if self.train:
            return self._getitem_train_(index)
        else:
            return self._getitem_test_(index)
    
    def _getitem_test_(self, index): 
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
        
    def _getitem_train_(self, index):
        start, end = self.video_indices[index]
        pose = self.labels['pose'][start:end+1]
        transl = self.labels['transl'][start:end+1]
        betas = self.labels['betas'][start:end+1]
        gt_kp3d = self.labels['gt_kp3d'][start:end+1]
        
        # Video data
        kp3d = self.labels['kp3d'][start:end+1]
        kp3d = augment_keypoints(kp3d, 'COCO17', 5e-3)
        
        # IMU data
        acc = self.labels['acc'][start:end+1]
        gyr = self.labels['gyr'][start:end+1]
        gyr, acc = augment_IMU(gyr, acc, 0.1)
        
        return {
            'pose': pose[:, cfg.SMPL.MAIN_JOINTS],
            'transl': transl,
            'betas': betas,
            'acc': acc / 9.81,
            'gyr': gyr / 10,
            'kp3d': kp3d,
            'gt_kp3d': gt_kp3d,
            'vid': self.labels['vid'][start]
        }
        
    def prepare_video_batch(self, step_size=None):
        self.video_indices = []
        video_names_unique, group = np.unique(
            self.labels['vid'], return_index=True)
        perm = np.argsort(group)
        self.video_names_unique, self.group = video_names_unique[perm], group[perm]
        indices = np.split(np.arange(0, self.labels['vid'].shape[0]), self.group[1:])

        if step_size is None:
            step_size = (self.args.seq_length) // 5

        for idx in range(len(video_names_unique)):
            indexes = indices[idx]
            if indexes.shape[0] < self.args.seq_length:
                continue
            chunks = view_as_windows(
                indexes, (self.args.seq_length), step=step_size)
            chunks = chunks[np.random.randint(5)::5]
            start_finish = chunks[:, (0, -1)].tolist()
            self.video_indices += start_finish
        
        
def get_totalcapture_dloader(args, train=False):
    print('Load Total Capture Dataset...')
    
    dloader = DataLoader(TotalCapture(args, train),
                         batch_size=args.batch_size if train else 1,
                         shuffle=train,
                         num_workers=args.num_workers,
                         pin_memory=args.pin_memory,
                         collate_fn=make_collate_fn())
    
    return dloader