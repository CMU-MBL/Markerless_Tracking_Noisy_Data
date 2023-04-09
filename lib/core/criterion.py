import torch
import numpy as np
from torch.nn import functional as F

from configs import configs as cfg
from lib.utils import transforms

from lib.utils.pose_utils import compute_similarity_transform_batch



class Criterion():
    def __init__(self, args):

        self.lw_pose = args.lw_pose
        self.lw_betas = args.lw_betas
        self.lw_transl = args.lw_transl
        self.lw_kp3d = args.lw_kp3d

        self.joint_type = args.joint_type


    def __call__(self, pred_smpl, pred_kp3d, groundtruths):
        loss_dict = {}

        pred_pose, pred_betas, pred_transl = pred_smpl
        pose, betas, transl, kp3d = [groundtruths[k] for k in ['pose', 'betas', 'transl', 'kp3d']]

        if self.lw_pose > 0 and pred_pose is not None:
            loss_dict = self._get_pose_loss(pose, pred_pose, loss_dict)
        if self.lw_betas > 0 and pred_betas is not None:
            loss_dict = self._get_shape_loss(betas, pred_betas, loss_dict)
        if self.lw_transl > 0 and pred_transl is not None:
            loss_dict = self._get_transl_loss(transl, pred_transl, loss_dict)
        if self.lw_kp3d > 0 and pred_kp3d is not None:
            loss_dict = self._get_kp3d_loss(kp3d, pred_kp3d, loss_dict)

        total_loss = 0
        for key in loss_dict.keys():
            total_loss += loss_dict[key]
            loss_dict[key] = loss_dict[key].item()

        if pred_kp3d is not None:
            loss_dict['MPJPE'] = self._get_mpjpe(kp3d, pred_kp3d)
        return total_loss, loss_dict


    def _get_pose_loss(self, pose, pred_pose, loss_dict):
        """Calculate SMPL pose loss using rotation matrix space"""

        pose_diff = pose.transpose(-1, -2) @ pred_pose
        pose_diff = torch.abs(transforms.rotation_matrix_to_angle_axis(pose_diff.reshape(-1, 3, 3)))
        pose_diff = (pose_diff ** 2).sum(-1).mean()
        loss = pose_diff * self.lw_pose
        loss_dict['Pose Loss'] = loss
        return loss_dict

    def _get_shape_loss(self, betas, pred_betas, loss_dict):
        """Calculate SMPL betas loss"""
        loss = F.mse_loss(betas, pred_betas, reduction='mean') * self.lw_betas
        loss_dict['Shape Loss'] = loss
        return loss_dict

    def _get_transl_loss(self, transl, pred_transl, loss_dict):
        """Calculate SMPL transl loss"""
        loss = F.mse_loss(transl, pred_transl, reduction='mean') * self.lw_transl
        loss_dict['Transl Loss'] = loss
        return loss_dict

    def _get_shape_const_loss(self, pred_betas, loss_dict):
        """Calculate shape variation loss"""
        loss = pred_betas.var(1).mean(0).sum()
        loss_dict['Const Loss'] = loss * self.lw_const
        return loss_dict

    def _get_kp3d_loss(self, kp3d, pred_kp3d, loss_dict):
        """Calculate keypoints loss"""
        loss = F.mse_loss(kp3d, pred_kp3d, reduction='mean') * self.lw_kp3d
        loss_dict['Kp 3d Loss'] = loss
        return loss_dict

    def _get_mpjpe(self, kp3d, pred_kp3d, align=False, return_batch=False):
        if align:
            pelv_idx = cfg.KEYPOINTS.PELVIS_IDX[self.joint_type]
            if len(pelv_idx) == 2:
                pelv = kp3d[..., pelv_idx, :].mean(-2, keepdims=True)
                pred_pelv = pred_kp3d[..., pelv_idx, :].mean(-2, keepdims=True)
            else:
                pelv = kp3d[..., pelv_idx:pelv_idx+1, :]
                pred_pelv = pred_kp3d[..., pelv_idx:pelv_idx+1, :]
            kp3d = kp3d - pelv
            pred_kp3d = pred_kp3d - pred_pelv

        jpe = torch.sqrt(torch.sum(torch.square(kp3d - pred_kp3d), -1))
        jpe = jpe.mean((1, 2)) * 1e3
        if return_batch: return jpe.detach()

        mpjpe = jpe.mean().detach()
        return mpjpe.item()

    def _get_pa_mpjpe(self, kp3d, pred_kp3d, return_batch=False):
        b, f, j = kp3d.shape[:3]
        kp3d = kp3d.detach().cpu().numpy().reshape(b * f, j, 3)
        pred_kp3d = pred_kp3d.detach().cpu().numpy().reshape(b * f, j, 3)

        pred_kp3d = compute_similarity_transform_batch(pred_kp3d, kp3d)
        kp3d = kp3d.reshape(b, f, j, 3)
        pred_kp3d = pred_kp3d.reshape(b, f, j, 3)

        jpe = np.sqrt(np.sum(np.square(kp3d - pred_kp3d), -1))
        jpe = jpe.mean((1, 2)) * 1e3
        if return_batch: return jpe

        mpjpe = jpe.mean()
        return mpjpe


    def _get_mpjae(self, pose, pred_pose, full_pose=False, reduction='mean'):
        """ Calculate Mean Per Joint Angle Error
        Args:
            pose: Groundtruth joint angle in 6D format, torch.Tensor (B, F, , 22, 3 3)
            pred_pose: Predicted joint angle in 6D format, torch.Tensor (B, F, 132)
            full_pose: Return joint angle error for all joints (lower limb only if False)

        Return mpjae - Joint angle error
        """

        B, F, J = pose.shape[:3]
        angle = transforms.rotation_matrix_to_angle_axis(pose.reshape(-1, 3, 3)).reshape(B, F, J, 3)
        pred_angle = transforms.rotation_matrix_to_angle_axis(pred_pose.reshape(-1, 3, 3)).reshape(B, F, J, 3)

        # error in shape B, F, J, 3
        error = torch.abs(angle - pred_angle) * 180 / np.pi

        # retreat uncontinuity issue
        error[error>180] = (360 - error[error>180])

        # Root Mean Square Error
        rmse = torch.sqrt((error ** 2).mean(1))

        if not full_pose:
            rmse = rmse[:, [1, 2, 4, 5, 7, 8]]

        if reduction == 'mean': rmse = rmse.mean((1, 2))
        return rmse
