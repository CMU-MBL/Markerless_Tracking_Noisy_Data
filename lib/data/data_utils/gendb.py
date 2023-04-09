import os.path as osp
from glob import glob

import torch
import numpy as np

from configs import configs as cfg
from lib.utils import transforms
from lib.data.data_utils.gendb_utils import (get_rmse,
                                             sync_data,
                                             read_calibration,
                                             parse_sensor_6axis,
                                             parse_calib_imu_ref,
                                             triangulate_keypoints)


smpl_to_sensor_idx_list = [15, 9, 0, 16, 17, 18, 19, 1, 2, 4, 5, 7, 8]
sensor_names = ('Head', 'Sternum', 'Pelvis', 'L_UpArm', 'R_UpArm', 'L_LowArm', 'R_LowArm',
              'L_UpLeg', 'R_UpLeg', 'L_LowLeg', 'R_LowLeg', 'L_Foot', 'R_Foot')  # imu bones

def process_imu_data(smpl, output, pose, subj, action, winsize=7, viz=False):
    """Process IMU data to align with SMPL joints. Then generate synthetic IMU data.
    """
    
    # Load IMU data
    calib_ref = parse_calib_imu_ref(
        osp.join(cfg.PATHS.TC_BASE, subj.upper(), f'{subj}_{action}_calib_imu_ref.txt'))
    imu_data = parse_sensor_6axis(
        osp.join(cfg.PATHS.TC_BASE, subj.upper(), f'{action}_Xsens_AuxFields.sensors'))
    
    min_len = min(len(pose), len(imu_data[0]['Head']))
    
    # TotalCapture Tracking frame to AMASS Tracking frame
    R_TA = torch.tensor([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]).float()
    
    # Compute global SMPL orientation from MoSh
    glob_pose = smpl.get_glob_pose(pose)[:min_len]
    
    R_Ai_list, R_is_list, A_i_list, G_i_list = [], [], [], []
    for smpl_idx, key in zip(smpl_to_sensor_idx_list, sensor_names):
        
        # Load IMU data
        Q_Ii = imu_data[0][key][:min_len]
        
        R_Ii = transforms.quaternion_to_matrix(Q_Ii)
        # Load reference to tracking frame calibration
        Q_TI = calib_ref[key]
        R_TI = transforms.quaternion_to_matrix(torch.Tensor(Q_TI))
        
        # Map IMU frame to AMASS tracking frame frame
        R_Ai = R_TA.T.view(1, 3, 3) @ R_TI.view(1, 3, 3) @ R_Ii
        
        # Compute calibration between IMU and SMPL in current frame
        R_As = glob_pose[:, smpl_idx]
        R_is = R_Ai.transpose(-1, -2) @ R_As
        
        # Load acceleration data
        A_i = torch.Tensor(imu_data[1][key])[:min_len]
        
        # Load angular velocity
        G_i = torch.Tensor(imu_data[2][key])[:min_len]
        
        # Append data
        R_Ai_list.append(R_Ai)
        R_is_list.append(R_is)
        A_i_list.append(A_i)
        G_i_list.append(G_i)
    
    # Concatenate data
    R_Ai_list = torch.stack(R_Ai_list, dim=0)
    R_is_list = torch.stack(R_is_list, dim=0)
    A_i_list = torch.stack(A_i_list, dim=0)
    G_i_list = torch.stack(G_i_list, dim=0)
    
    # Get mean calibration across frames
    R_is = transforms.rotation_6d_to_matrix(
        transforms.matrix_to_rotation_6d(R_is_list).mean(dim=1, keepdims=True)
    )
    
    # Align IMU to SMPL coordinate based on mean calibration
    R_As_ = (R_Ai_list @ R_is).permute(1, 0, 2, 3)
    A_s = (R_is.transpose(-1, -2) @ A_i_list.unsqueeze(-1)).squeeze(-1).permute(1, 0, 2)
    G_s = (R_is.transpose(-1, -2) @ G_i_list.unsqueeze(-1)).squeeze(-1).permute(1, 0, 2)
    
    # Compute virtual IMU data from SMPL
    # Total Capture has framerate of 60Hz
    A_s_virt = smpl.get_acceleration(pose, output, fps=60)[:min_len]
    G_s_virt = smpl.get_angular_velocity(pose, fps=60)[:min_len]
    
    sync_frame = sync_data(A_s, A_s_virt)
    A_s_virt = torch.roll(A_s_virt, shifts=sync_frame, dims=0)
    G_s_virt = torch.roll(G_s_virt, shifts=sync_frame, dims=0)
    
    # Visualize check for the acceleration and gyro data
    if viz:
        import os
        import matplotlib.pyplot as plt
        os.makedirs(osp.join('output', 'data_processing', 'IMU_comparison', subj, action), exist_ok=True)
        for s_idx in range(13):
            _, ax = plt.subplots(1, 2, figsize=(10, 5))
            for a_idx, linestyle in enumerate(['-', '--', '-.']):
                ax[0].plot(A_s[1000:1500, s_idx, a_idx], color='tab:red', linestyle=linestyle)
                ax[0].plot(A_s_virt[1000:1500, s_idx, a_idx], color='dimgrey', linestyle=linestyle)
                ax[1].plot(G_s[1000:1500, s_idx, a_idx], color='tab:red', linestyle=linestyle)
                ax[1].plot(G_s_virt[1000:1500, s_idx, a_idx], color='dimgrey', linestyle=linestyle)
            plt.savefig(osp.join(
                'output', 'data_processing', 'IMU_comparison', 
                subj, action, f'imu_{s_idx}.png')
                )
            plt.close()
    
    rmse_A = get_rmse(A_s, A_s_virt)
    rmse_G = get_rmse(G_s, G_s_virt)
    
    return R_As_, A_s, G_s, A_s_virt, G_s_virt, sync_frame, rmse_A, rmse_G


def process_video_data(joints, subj, action, viz=False):
    # Read camera calibration
    # calibs = torch.load(cfg.PATHS.TC_CALIB)['cameras']
    # R = transforms.rotation_6d_to_matrix(calibs['r6d'].cpu()).numpy()
    calibs = read_calibration(cfg.PATHS.TC_CALIB)
    R = calibs['R'].cpu().numpy()
    T = calibs['T'].cpu().numpy()
    K = calibs['K'].cpu().numpy()
    
    detection_file_list = glob(osp.join(cfg.PATHS.TC_DETECTION, f'{subj.upper()}_{action}_*.npy'))
    detection_file_list.sort()
    
    assert len(detection_file_list) == len(R)
    
    select_cams = [0, 3, 4, 7]
    detection_list = []
    len_list = [len(joints)]
    for idx, detection_file in enumerate(detection_file_list):
        if idx in select_cams:
            detection = np.load(detection_file)
            detection_list.append(detection)
            len_list.append(len(detection))
        
    detection_list = np.stack(
        [detection[:min(len_list)] for detection in detection_list]
    )
    joints = joints[:min(len_list)]
    
    # Triangulate 2D detection
    detection_3d = triangulate_keypoints(detection_list, K, R, T, select_cams)
        
    if viz:
        import os, cv2, imageio
        from lib.utils.viz import render_keypoints_on_image
        
        os.makedirs('output/data_processing/keypoints', exist_ok=True )
        
        # By projecting on cam8 (which has the largest FoV), check the quality
        vidname = f'dataset/TotalCapture_videos/input_video/TC_{subj.upper()}_{action}_cam8.mp4'
        vidcap = cv2.VideoCapture(vidname)
        _R = torch.from_numpy(R[-1]).float()
        _T = torch.from_numpy(T[-1]).float()
        _K = torch.from_numpy(K[-1]).float()
        
        frame = 0
        writer = imageio.get_writer(
            f'output/data_processing/keypoints/{subj}_{action}.mp4',
            fps=10, macro_block_size=None)
        while frame < 3000:
            ret, image = vidcap.read()
            if not ret: break
            if frame >= min(len_list): break
            if frame % 20 == 0:
                image = render_keypoints_on_image(joints[frame], image, _R, _T, _K, 'coco', color=(0, 0, 255))
                image = render_keypoints_on_image(detection_3d[frame], image, _R, _T, _K, 'coco', color=(0, 255, 255))
                writer.append_data(image[..., ::-1])
            frame += 1
        writer.close()
    
    # Compute RMSE
    rmse = get_rmse(joints, detection_3d, True) * 1e3
    
    return detection_3d, rmse