import re
from collections import deque, defaultdict

import torch
import numpy as np
from torch.nn import functional as F


bone_names = ('Head', 'Sternum', 'Pelvis', 'L_UpArm', 'R_UpArm', 'L_LowArm', 'R_LowArm',
              'L_UpLeg', 'R_UpLeg', 'L_LowLeg', 'R_LowLeg', 'L_Foot', 'R_Foot')  # imu bones

# re
float_reg = r'(\-?\d+\.\d+)'
int_reg = r'[\d]+'
word_reg = r'[a-zA-z\_]+'

float_finder = re.compile(float_reg)
int_finder = re.compile(int_reg)
word_finder = re.compile(word_reg)

def read_calibration(filename, calib_to_SMPL=True):
    import torch
    from collections import deque

    # TotalCapture and AMASS have different coordinate system
    transform = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])

    with open(filename, 'r') as calib:
        lines = calib.readlines()

    lines = deque(lines)
    num_cameras, dist_order = lines.popleft().split()
    num_cameras, dist_order = int(num_cameras), int(dist_order)
    cameras = []

    for i in range(num_cameras):
        min_row, max_row, min_col, max_col = tuple(map(int, lines.popleft().split()))
        fx, fy, cx, cy = tuple(map(float, lines.popleft().split()))
        dist_param = float(lines.popleft())
        # dist_param *= 1e4
        r1 = list(map(float, lines.popleft().split()))
        r2 = list(map(float, lines.popleft().split()))
        r3 = list(map(float, lines.popleft().split()))
        R = np.array([r1, r2, r3])
        if calib_to_SMPL:
            R = R @ transform
        t = np.reshape(np.array(list(map(float, lines.popleft().split()))), (3, 1))
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        camera = {'R': torch.from_numpy(R).float(), 'T': torch.from_numpy(t).float(),
                  'K': torch.from_numpy(K).float(), 'dist': dist_param}
        cameras += [camera]

    out_dict = dict()
    for key in ['R', 'T', 'K']:
        out_dict[key] = torch.stack([camera[key] for camera in cameras])
    return out_dict

def parse_sensor_6axis(fpath):
    with open(fpath, 'r') as f:
        lines = f.readlines()
    lines = deque(lines)

    seq_meta = lines.popleft()
    seq_meta_result = int_finder.findall(seq_meta)
    assert len(seq_meta_result) == 2, 'error in seq meta data'
    num_sensors = int(seq_meta_result[0])
    num_frames = int(seq_meta_result[1])

    ori, acc, gyr = defaultdict(list), defaultdict(list), defaultdict(list)
    for f in range(num_frames):
        # frame
        frame_index_line = lines.popleft()
        frame_index = int(int_finder.findall(frame_index_line)[0])

        for _ in range(num_sensors):
            onejoint = lines.popleft()
            joint_name = word_finder.findall(onejoint)[0]
            assert joint_name in bone_names, 'invalid joint name: {} in frame {}'.format(joint_name, frame_index)
            values = float_finder.findall(onejoint)
            orientation = tuple(float(x) for x in values[0:4])
            acceleration = tuple(float(x) for x in values[4:7])
            gyros = tuple(float(x) for x in values[7:10])
            # joints[joint_name] = (orientation, acceleration, gyros)
            ori[joint_name].append(orientation)
            acc[joint_name].append(acceleration)
            gyr[joint_name].append(gyros)
    
    for key in ori.keys():
        ori[key] = torch.from_numpy(np.array(ori[key])).float()
        acc[key] = torch.from_numpy(np.array(acc[key])).float()
        gyr[key] = torch.from_numpy(np.array(gyr[key])).float()
    
    return (ori, acc, gyr)


def parse_calib_imu_ref(fpath):
    with open(fpath, 'r') as f:
        lines = f.readlines()
    lines = deque(lines)
    ref_num_sensors = int(lines.popleft())
    assert ref_num_sensors == len(bone_names), 'mismatching sensor nums with ref sensor nums'
    ref_joints = dict()
    for i in range(ref_num_sensors):
        onejoint = lines.popleft()
        joint_name = word_finder.findall(onejoint)[0]
        assert joint_name in bone_names, 'invalid joint name: {}'.format(joint_name)
        values = float_finder.findall(onejoint)
        assert len(values) == 4, 'wrong number of joint parameter'
        # orientation in ref is ordered as (x y z w), which is different from captured data (w x y z)
        orientation_imag = [float(x) for x in values[0:3]]
        orientation = [float(values[3])]
        orientation.extend(orientation_imag)
        ref_joints[joint_name] = orientation
    return ref_joints


def sync_data(data1, data2, winsize=7):
    start, end = 500, -500
    
    rmse_list = []
    for i in range(-winsize, winsize+1):
        rmse = torch.mean(torch.sqrt(torch.sum(torch.square(
            data1[start+i:end+i] - data2[start:end]), dim=-1)))
        rmse_list.append(rmse)
        
    sync_frame = -winsize + np.argmin(np.array(rmse_list))
    return sync_frame
    


def convolution_filtering(signal, winsize):
    f, n, d = signal.shape
    signal = signal.reshape(f, n * d).T.unsqueeze(1)
    
    window = torch.hann_window(winsize, dtype=torch.float32, device=signal.device)
    filter = window / window.sum()
    
    signal = F.conv1d(signal, filter.unsqueeze(0).unsqueeze(0), padding=window.shape[0] // 2)
    signal = signal.permute(1, 2, 0).reshape(f, n, d)
    return signal


def triangulate_keypoints(keypoints, Ks, Rs, Ts, camera_ids):
    """
    Triangulate multiple 2D detection using camera matrices.

    Args:
        keypoints (numpy array): 2D detection from multiple cameras. The shape is (num_cameras, num_frames, num_joints, 3)
        Ks (numpy array): camera intrinsic parameters. The shape is (num_total_cameras, 3, 3)
        Rs (numpy array): camera rotation parameters. The shape is (num_total_cameras, 3, 3)
        Ts (numpy array): camera translation parameters. The shape is (num_total_cameras, 3)
        camera_ids (list): the indices of target cameras to use for triangulation.

    Returns:
        triangulated_keypoints (numpy array): triangulated keypoints. The shape is (num_frames, num_joints, 3).
    """

    n_c, n_f, n_j = keypoints.shape[:3]
    triangulated_keypoints = np.zeros((n_f, n_j, 3))

    projs = []
    for camera_id in camera_ids:
        E = np.hstack([Rs[camera_id], Ts[camera_id].reshape(3, 1)])
        proj = Ks[camera_id].dot(E)
        projs += [proj]
    projs = np.stack(projs)
    
    for f in range(n_f):
        for j in range(n_j):
            x2d = keypoints[:, f, j, :2]
            conf = keypoints[:, f, j, -1] ** 2      # 
            n_available_cameras = (conf > 0).sum()
            # average_conf = (conf[conf > 0]).sum() / n_c
            
            if n_available_cameras < 2:# or average_conf < 0.2: 
                continue
            conf = np.clip(conf, 0, 1)
            
            A = np.repeat(projs[:, 2:3], 2, 1) * x2d.reshape(n_c, 2, 1)
            A -= projs[:, :2]
            A *= conf.reshape(-1, 1, 1)
            
            (u, s, vh) = np.linalg.svd(A.reshape(-1, 4), full_matrices=False)
            x3d_hom = vh[3, :]
            x3d = x3d_hom[:-1] / x3d_hom[-1]
            triangulated_keypoints[f, j] = x3d
    
    return torch.from_numpy(triangulated_keypoints).float()


def get_rmse(data1, data2, align=False):
    if align:
        data1 = data1 - data1.mean(-2, keepdims=True)
        data2 = data2 - data2.mean(-2, keepdims=True)
    return torch.sqrt(torch.mean(torch.sum(torch.square(data1 - data2), dim=-1)))
    