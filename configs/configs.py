class CONSTANTS:
    FPS = 25
    # FPS = 30

class PATHS:
    TC_BASE = 'dataset/TotalCapture'
    # TC_CALIB = 'dataset/TotalCapture/calibration.pt'
    TC_CALIB = 'dataset/TotalCapture/calibration.cal'
    TC_DETECTION = 'dataset/TotalCapture/detection/cpn'
    
    AMASS_BASE = 'dataset/AMASS'
    SMPL = 'dataset/body_models/smpl'
    
    TC_LABEL = {
        'train': 'dataset/HumanPose/VIF/totalcapture_test_4cams_train.pt',
        'test': 'dataset/HumanPose/VIF/totalcapture_test_4cams_test.pt'
    }
    # AMASS_LABEL = 'dataset/HumanPose/VIF/amass_train.pt'
    AMASS_LABEL = 'dataset/HumanPose/VIF/amass_train_prev.pt'
    

class SMPL:
    JOINT_NAMES = [
        'Pelvis', 'Left Hip', 'Right Hip', 'Spine 1 (Lower)', 'Left Knee',
        'Right Knee', 'Spine 2 (Middle)', 'Left Ankle', 'Right Ankle',
        'Spine 3 (Upper)', 'Left Foot', 'Right Foot', 'Neck',
        'Left Shoulder (Inner)', 'Right Shoulder (Inner)', 'Head',
        'Left Shoulder (Outer)', 'Right Shoulder (Outer)', 'Left Elbow',
        'Right Elbow', 'Left Wrist', 'Right Wrist', 'Left Hand', 'Right Hand'
    ]
    
    JOINT_MAP = {
        'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
        'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
        'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
        'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
        'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
        'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
        'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
        'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
    }
    
    OP_JOINT_NAMES = [
        # 25 OpenPose joints (in the order provided by OpenPose)
        'OP Nose', 'OP Neck', 'OP RShoulder', 'OP RElbow',
        'OP RWrist', 'OP LShoulder', 'OP LElbow', 'OP LWrist',
        'OP MidHip', 'OP RHip', 'OP RKnee', 'OP RAnkle',
        'OP LHip', 'OP LKnee', 'OP LAnkle', 'OP REye',
        'OP LEye', 'OP REar', 'OP LEar', 'OP LBigToe',
        'OP LSmallToe', 'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',
    ]
    
    # MAIN_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    MAIN_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    REGRESSOR = 'dataset/body_models/J_regressor_coco.npy'

class IMU:
    LIST = [
        'Head', 'Sternum', 'Pelvis', 'L_UpArm', 'R_UpArm',
        'L_LowArm', 'R_LowArm', 'L_UpLeg', 'R_UpLeg',
        'L_LowLeg', 'R_LowLeg', 'L_Foot', 'R_Foot',
    ]
    
    TO_VERTS = {
        'Head': 411, 'Sternum': 3076, 'Pelvis': 3502, 'L_UpArm': 1379,
        'R_UpArm': 4849, 'L_LowArm': 1952, 'R_LowArm': 5422, 'L_UpLeg': 847,
        'R_UpLeg': 4712, 'L_LowLeg': 1373, 'R_LowLeg': 4561, 'L_Foot': 3345,
        'R_Foot': 6745
    }

    TO_SMPL_MAP = {
        'Head': 'Head', 'Sternum': 'Spine 3 (Upper)', 'Pelvis': 'Pelvis',
        'L_UpArm': 'Left Shoulder (Outer)', 'R_UpArm': 'Right Shoulder (Outer)',
        'L_LowArm': 'Left Elbow', 'R_LowArm': 'Right Elbow',
        'L_UpLeg': 'Left Hip', 'R_UpLeg': 'Right Hip',
        'L_LowLeg': 'Left Knee', 'R_LowLeg': 'Right Knee',
        'L_Foot': 'Left Ankle', 'R_Foot': 'Right Ankle'
    }
    
    SMPL_IDXS = [15, 9, 0, 16, 17, 18, 19, 1, 2, 4, 5, 7, 8]
    
class KEYPOINTS:
    PELVIS_IDX = {'TC16': 0, 'COCO17': [12, 11]}