from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys

import torch
import numpy as np
from smplx import SMPL as _SMPL
from smplx.utils import SMPLOutput as ModelOutput
from smplx.lbs import vertices2joints

from lib.utils import transforms
from configs import configs as cfg


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, J_regressor, *args, **kwargs):
        sys.stdout = open(os.devnull, 'w')
        super(SMPL, self).__init__(*args, **kwargs)
        sys.stdout = sys.__stdout__
        J_regressor_extra = np.load(J_regressor)
        self.register_buffer('J_regressor_extra', torch.tensor(
            J_regressor_extra, dtype=torch.float32))

        joints = [cfg.SMPL.JOINT_MAP[i] for i in cfg.SMPL.OP_JOINT_NAMES]
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def get_full_pose(self, reduced_pose):
        full_pose = torch.eye(
            3, device=reduced_pose.device
        )[(None, ) * 2].repeat(reduced_pose.shape[0], 24, 1, 1)
        full_pose[:, cfg.SMPL.MAIN_JOINTS] = reduced_pose
        return full_pose

    def forward(self, pred_smpl, groundtruths=None):
        pred_pose, betas, transl = pred_smpl
        if pred_pose.shape[-1] == 6: 
            pred_pose = transforms.rot6d_to_rotmat(pred_pose)
        pred_pose = pred_pose.reshape(-1, len(cfg.SMPL.MAIN_JOINTS), 3, 3)
        full_pose = self.get_full_pose(pred_pose)

        if betas is None:
            b = full_pose.shape[0]
            if groundtruths is not None:
                # Model type is IMU, use GT betas, transl and global orientation
                betas = groundtruths['betas'].view(b, -1)
                transl = groundtruths['transl'].view(b, -1)
                full_pose[:, :1] = groundtruths['pose'].view(b, -1, 3, 3)[:, :1]
            else:
                device = full_pose.device
                betas = torch.zeros((b, 10), dtype=torch.float32).to(device)
                transl = torch.zeros((b, 3), dtype=torch.float32).to(device)
                full_pose[:, :1] = torch.zeros((b, 1, 3, 3), dtype=torch.float32).to(device)
        
        output = self.get_output(body_pose=full_pose[:, 1:],
                                 global_orient=full_pose[:, :1],
                                 betas=betas.view(-1, 10),
                                 transl=transl.view(-1, 3),
                                 return_full_pose=True,
                                 pose2rot=False)
        return output

    def get_output(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        if self.J_regressor_extra is not None:
            joints = vertices2joints(self.J_regressor_extra,
                                     smpl_output.vertices)
        else:
            joints = smpl_output.joints[:, self.joint_map, :]

        output = ModelOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output
    
    
    def get_glob_pose(self, pose):
        results = []
        root_pose = pose[:, 0]
        results += [root_pose]
        for i in range(1, pose.size(1)):
            loc_pose = pose[:, i]
            global_pose = results[self.parents[i]] @ loc_pose
            results.append(global_pose)

        results = [result.unsqueeze(1) for result in results]
        return torch.cat(results, dim=1)
    
    
    def get_orientation(self, pose):
        glob_pose = self.get_glob_pose(pose)
        orientation = glob_pose[:, cfg.IMU.SMPL_IDXS]
        return orientation
    
    def get_angular_velocity(self, pose, fps=None):
        dtype = pose.dtype
        device = pose.device
        
        if fps is None: fps = cfg.CONSTANTS.FPS
        orientation = self.get_orientation(pose)
        results = []
        for idx in range(len(cfg.IMU.LIST)):
            tmp = torch.transpose(orientation[:-1, idx], 1, 2) @ orientation[1:, idx]
            tmp = transforms.rotation_matrix_to_angle_axis(tmp) * fps
            results.append(tmp.unsqueeze(1))
        gyr_ = torch.cat(results, dim=1)
        gyr = torch.zeros((gyr_.shape[0] + 1, gyr_.shape[1], 3)).float().to(
            device=device, dtype=dtype)
        gyr[:-1] = gyr_
        return gyr
    
    def get_acceleration(self, pose, output, fps=None):
        device = pose.device
        dtype = pose.dtype
        
        if fps is None: fps = cfg.CONSTANTS.FPS
        vertices = output.vertices
        orientation = self.get_orientation(pose)
        
        results = []
        for idx, sensor in enumerate(cfg.IMU.LIST):
            loc = vertices[:, cfg.IMU.TO_VERTS[sensor]]
            tmp = (loc[2:] + loc[:-2] - 2 * loc[1:-1]) * (fps ** 2)
            tmp[:, 2] += 9.81
            tmp = (orientation[1:-1, idx].transpose(-1, -2) @ tmp.unsqueeze(-1)).squeeze(-1)
            results.append(tmp.unsqueeze(1))
        acc_ = torch.cat(results, dim=1)
        acc = torch.zeros((acc_.shape[0] + 2, acc_.shape[1], 3)).float().to(
            device=device, dtype=dtype)
        acc[1:-1] = acc_
        return acc
