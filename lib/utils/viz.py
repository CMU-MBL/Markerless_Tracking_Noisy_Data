import cv2
import torch


# # Draw 2D keypoints
CONNECTIVITY = {
    'TotalCapture': [
        (0,5), (0,18), (0,15), (5,6), (5,8), (8,9), (9,10), (5,12),
        (12,13), (13,14), (18,19), (19,20), (15,16), (16, 17)],
    'TC16': [(0,1), (1,2), (2,3), (0,4), (4,5), (5,6), (0,7),
             (7,8), (8,13), (13,14), (14,15), (8,10), (10,11), (11,12),
             (8, 9)],
    'coco': [
        (0, 1), (1, 3), (0, 2), (2, 4), # face
        (0, 5), (0, 6), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (6, 12), (5, 11), # upper body
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # lower body
    ]}


def pose3d_to_pose2d(x3d, _R, _T, _K):
    loc3d = _R @ x3d.T + _T.view(3, 1)
    loc2d = torch.div(loc3d, loc3d[2])
    x2d = torch.matmul(_K, loc2d)[:2]
    x2d = x2d.T

    return x2d

def render_keypoints_on_image(keypoints3d, image, R, T, K, jtype, linewidth=5,
                              conf=None, put_text=False, color=None):
    """Render keypoints onto the image
    Args:
        keypoints3d: torch.Tensor, (J, 3)
        image: numpy.ndarray, (H, W, 3)
        R: torch.Tensor, (3, 3)
        T: torch.Tensor, (3, 1)
        K: torch.Tensor, (3, 3)
        options
    """

    connectivity = CONNECTIVITY[jtype]

    if keypoints3d.shape[-1] == 2:
        keypoints2d = keypoints3d.astype('int')
    else:
        keypoints2d = pose3d_to_pose2d(
            keypoints3d, R, T, K).numpy().astype('int')
    if conf is not None:
        keypoints2d[conf==0] = 0
    x, y = keypoints2d[:, 0], keypoints2d[:, 1]
    for index_set in connectivity:
        xs, ys = [], []
        for index in index_set:
            if (x[index] > 1e-5 and y[index] > 1e-5):
                xs.append(x[index])
                ys.append(y[index])
        if len(xs) == 2:
            # Draw line
            start = (xs[0], ys[0])
            end = (xs[1], ys[1])
            image = cv2.line(image, start, end, color, linewidth)

    return image