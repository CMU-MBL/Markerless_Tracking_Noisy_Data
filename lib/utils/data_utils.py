from collections import defaultdict

import torch
import numpy as np

from configs import configs as cfg
from lib.utils import transforms
from lib.models.builder import build_smpl


def make_collate_fn():
    def collate_fn(items):
        items = list(filter(lambda x: x is not None , items))
        batch = dict()
        batch['vid'] = [item['vid'] for item in items]
        for key in items[0].keys():
            try: batch[key] = torch.stack([item[key] for item in items])
            except: pass
        return batch

    return collate_fn


def prepare_batch(args, batch):
    inputs = defaultdict()
    groundtruths = defaultdict()
    
    try:
        groundtruths['pose'] = batch['pose'].to(device=args.device)
    except: import pdb; pdb.set_trace()
    groundtruths['betas'] = batch['betas'].to(device=args.device)
    groundtruths['kp3d'] = batch['gt_kp3d'].to(device=args.device)
    groundtruths['transl'] = batch['transl'].to(device=args.device)

    try:
        inputs['kp3d'] = batch['kp3d'].to(device=args.device)
    except: pass

    try:
        gyr = batch['gyr'].to(device=args.device)
        acc = batch['acc'].to(device=args.device)
        inputs['imu'] = torch.cat((gyr, acc), dim=-1)
    except: pass

    return inputs, groundtruths


_smpl = None
def get_smpl(args):
    global _smpl
    if _smpl is None:
        _smpl = build_smpl(args, device='cpu', batch_size=args.seq_length+2)
    return _smpl

_protocol = None
def get_protocol():
    global _protocol
    if _protocol is None:
        protocol = np.load('dataset/protocol.npy', allow_pickle=True).item()
        _protocol = []
        for key in protocol:
            _protocol.append(torch.from_numpy(protocol[key]).float())
        _protocol = torch.stack(_protocol)
    return _protocol


def get_smpl_output(pose, betas, transl):
    _smpl_output = _smpl.get_output(
        body_pose=pose[:, 1:],
        global_orient=pose[:, :1],
        betas=betas,
        transl=transl,
        pose2rot=False)
    return _smpl_output


def get_keypoints_noise(joint_type, noise_std):
    if joint_type == 'COCO17':
        noise_factor = torch.tensor([3, 2, 2, 2, 2, # Face
                                     5, 5, 3, 3, 2, 2,  # Arms
                                     5, 5, 3, 3, 2, 2, # Legs
                                     ]).unsqueeze(-1).expand(17, 3)
        peak_factor = torch.tensor([1, 1, 1, 1, 1, # Face
                                    2, 2, 5, 5, 3, 3, # Arms
                                    1, 1, 5, 5, 5, 5, # Legs
                                    ]).unsqueeze(-1).expand(17, 3)
    elif joint_type == 'TC16':
        noise_factor = torch.tensor([2, 5, 3, 2, 5, 3, 2, 5, 4, 3, 5, 3, 2,
                                        5, 3, 2]).unsqueeze(-1).expand(16, 3)
        peak_factor = torch.tensor([0, 1, 5, 5, 1, 5, 5, 0.2, 0.5, 1, 2, 5,
                                    3, 2, 5, 3]).unsqueeze(-1).expand(16, 3)
    noise_factor = noise_factor.float() * noise_std

    return noise_factor, peak_factor


def get_synthetic_keypoints(output, joint_type, noise_std):
    """Get 3D keypoints of the motion"""
    gt_kp3d = output.joints.clone().detach()
    input_kp3d = augment_keypoints(gt_kp3d.clone(), joint_type, noise_std)
    return gt_kp3d, input_kp3d


def get_synthetic_IMU(pose, output, noise_std):
    """Get synthetic IMU data of the motion"""
    
    gyr = _smpl.get_angular_velocity(pose)
    acc = _smpl.get_acceleration(pose, output)
    
    gyr, acc = augment_IMU(gyr, acc, noise_std)
    return gyr, acc


def augment_IMU(gyr, acc, noise_std):
    """Currently only implemented random noise for calibration:
    TODO: Update code to rotate more in axial direction
    """
    protocol = get_protocol()
    noise = torch.normal(mean=torch.zeros((len(cfg.IMU.LIST), 3)),
                        std=torch.ones((len(cfg.IMU.LIST), 3)) * noise_std)
    
    # Get IMU miscalibration error
    for idx, key in enumerate(cfg.IMU.LIST):
        if key in ['L_UpArm', 'L_LowArm', 'R_UpArm', 'R_LowArm',
                    'L_UpLeg', 'L_LowLeg', 'R_UpLeg', 'R_LowLeg',
                    'L_Foot', 'R_Foot']:
            noise[idx, 0] *= 3
        # Shoulder and foot segments are difficult to place IMU
        if key in ['L_UpArm', 'R_UpArm', 'L_Foot', 'R_Foot']:
            noise[idx, [1, 2]] *= 3
    
    noise = transforms.angle_axis_to_rotation_matrix(noise)
    noise = protocol @ noise
    
    # Transform IMU data based on the miscalibration error
    gyr = (noise.unsqueeze(0).transpose(-1, -2) @ gyr.unsqueeze(-1)).squeeze(-1)
    acc = (noise.unsqueeze(0).transpose(-1, -2) @ acc.unsqueeze(-1)).squeeze(-1)
    
    return gyr, acc
    

def augment_smpl_params(pose, transl, betas):
    """Agument SMPL parameters"""
    
    seq_length = pose.shape[0]
    
    # Random rotation in Z-axis (vertical to ground)
    angle = torch.rand(1) * 2 * np.pi
    euler = torch.tensor([0, 0, angle]).float().unsqueeze(0)        # Now Z-axis is vertical to the ground
    rmat = transforms.angle_axis_to_rotation_matrix(euler)
    pose[:, 0] = rmat @ pose[:, 0]
    transl = (rmat @ transl.T).squeeze().T
    
    # Random shape variability
    shape_noise = torch.normal(mean=torch.zeros((1, 10)),
                    std=torch.ones((1, 10))).expand(seq_length, 10)
    betas = betas + shape_noise
    
    return pose, transl, betas


def augment_keypoints(kp3d, joint_type, noise_std):
    """Augment 3D keypoints"""
    
    seq_length = kp3d.shape[0]
    noise_factor, peak_factor = get_keypoints_noise(joint_type, noise_std)

    # Noise type 1. Time invariant bias (maybe due to cloth)
    t_invariant_noise = torch.normal(
        mean=torch.zeros((len(noise_factor), 3)), std=noise_factor
    ).unsqueeze(0).expand(seq_length, len(noise_factor), 3)

    # Noise type 2. High frequency jittering noise
    t_variant_noise = torch.normal(
        mean=torch.zeros((seq_length, len(noise_factor), 3)),
        std=noise_factor.unsqueeze(0).expand(
            seq_length, len(noise_factor), 3)/3)

    # Noise type 3. Low frequency high magnitude noise
    peak_noise_mask = (torch.rand(seq_length, noise_factor.size(0)) < 2e-2
                        ).float().unsqueeze(-1).repeat(1, 1, 3)
    peak_noise = peak_noise_mask * torch.randn(3) * noise_std
    peak_noise = peak_noise * peak_factor

    # Confidence (only to set the random same)
    noise = t_invariant_noise + t_variant_noise + peak_noise
    conf_randomizer = torch.rand(*noise.shape) * 3 + 15
    conf = torch.exp(-torch.abs(noise)*conf_randomizer).mean(-1, keepdims=True)

    # Calculate loss and augment keypoints
    kp3d += noise

    return kp3d