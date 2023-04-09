import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage.util.shape import view_as_windows

from configs import configs as cfg
from lib.utils import transforms
from lib.utils.data_utils import (get_smpl,
                                  make_collate_fn,
                                  get_smpl_output,
                                  get_synthetic_IMU,
                                  get_synthetic_keypoints,
                                  augment_smpl_params,
)


class AMASS(Dataset):
    def __init__(self, args):
        import joblib
        self.model_type = args.model_type
        self.seq_length = args.seq_length
        self.joint_type = args.joint_type
        self.labels = joblib.load(cfg.PATHS.AMASS_LABEL)
        
        self.prepare_video_batch()
        print(f'AMASS dataset number of videos: {len(self.video_indices)}')
        
        get_smpl(args)
        self.reset_calib_noise(std=args.calib_noise)
        self.reset_keypoints_noise(std=args.keypoints_noise)
        
    def __len__(self):
        return len(self.video_indices)

    def __getitem__(self, index):
        return self.get_single_item(index)
        
    def prepare_video_batch(self, step_size=None):
        self.video_indices = []
        video_names_unique, group = np.unique(
            self.labels['vid_name'], return_index=True)
        perm = np.argsort(group)
        self.video_names_unique, self.group = video_names_unique[perm], group[perm]
        indices = np.split(np.arange(0, self.labels['vid_name'].shape[0]), self.group[1:])

        if step_size is None:
            step_size = (self.seq_length + 2) // 5

        for idx in range(len(video_names_unique)):
            indexes = indices[idx]
            if indexes.shape[0] < self.seq_length + 2:
                continue
            chunks = view_as_windows(
                indexes, (self.seq_length + 2), step=step_size)
            chunks = chunks[np.random.randint(5)::5]
            start_finish = chunks[:, (0, -1)].tolist()
            self.video_indices += start_finish
            
    def reset_calib_noise(self, std=1.5e-1):
        self.calib_noise = std

    def reset_keypoints_noise(self, std=5e-3):
        self.keypoints_noise = std
        
    def get_single_item(self, index):
        start_index, end_index = self.video_indices[index]
        pose = self.labels['pose'][start_index:end_index+1]
        pose = torch.from_numpy(pose).reshape(-1, 3).float()
        pose = transforms.angle_axis_to_rotation_matrix(pose).reshape(-1, 24, 3, 3)
        transl = torch.from_numpy(self.labels['transl'][start_index:end_index+1]).float()
        betas = torch.from_numpy(self.labels['betas'][start_index:end_index+1]).float()
        
        # Augment SMPL parameters
        pose, transl, betas = augment_smpl_params(pose, transl, betas)
        
        batch = {}
        batch['pose'] = pose[1:-1, cfg.SMPL.MAIN_JOINTS]
        batch['transl'] = transl[1:-1]
        batch['betas'] = betas[1:-1]
        batch['vid'] = self.labels['vid_name'][start_index]
        
        # Get SMPL output
        output = get_smpl_output(pose, betas, transl)
        
        # Get synthetic keypoints data
        gt_kp3d, input_kp3d = get_synthetic_keypoints(output, self.joint_type, self.keypoints_noise)
        batch['gt_kp3d'] = gt_kp3d[1:-1]; batch['kp3d'] = input_kp3d[1:-1]

        # Get synthetic IMU data
        if self.model_type in ['fusion', 'imu']:
            gyr, acc = get_synthetic_IMU(pose, output, self.calib_noise)
            batch['acc'] = acc[1:-1] / 9.81
            batch['gyr'] = gyr[1:-1] / 10
        
        return batch


def get_amass_dloader(args):
    print('Load AMASS Dataset...')
    dloader = DataLoader(AMASS(args),
                         batch_size=args.batch_size,
                         shuffle=True,
                         num_workers=args.num_workers,
                         pin_memory=args.pin_memory,
                         collate_fn=make_collate_fn())
    
    return dloader