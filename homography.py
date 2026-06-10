import numpy as np
import cv2
import calibration

def get_attitude(pose):
    # read IMU telemetry here
    # pose = [x,y,z,Y,P,R] orientation from the IMU/GP drone metadata. Angles in radians

    Y = pose[3] # yaw (z)
    P = pose[4] # pitch (y)
    R = pose[5] # roll (x)

    attitude = [Y,P,R]
    return attitude

def inverse_rotation_homography(attitude, K):
    # K = camera intrinsics calculated from chessboard image in calibration.py 
    
    Y = attitude[0]
    P = attitude[1]
    R = attitude[2]

    Rx = np.array([[1,         0,            0],
                   [0, np.cos(R), -1*np.sin(R)],
                   [0, np.sin(R),    np.cos(R)]])
    
    Ry = np.array([[   np.cos(P), 0, np.sin(P)],
                   [           0, 1,        0,],
                   [-1*np.sin(P), 0, np.cos(P)]])
    
    Rz = np.array([[np.cos(Y), -1*np.sin(Y), 0],
                   [np.sin(Y),    np.cos(Y), 0],
                   [        0,            0, 1]])
    
    R = Rz @ Ry @ Rx # total rotation
    K_inv = np.linalg.inv(K)
    H = K @ R.T @ K_inv 
    return H

def unrotate_image(img, pose, K):

    attitude = get_attitude(pose)
    H = inverse_rotation_homography(attitude, K)
    h, w = img.shape[:2]
    
    corners = np.float32([[[0,0]], [[0,h]], [[w,h]], [[w,0]]])
    transformed_corners = cv2.perspectiveTransform(corners, H)
    xs = transformed_corners[:, 0, 0]
    ys = transformed_corners[:, 0, 1]
    x_min, y_min = xs.min(), ys.min()
    x_max, y_max = xs.max(), ys.max()

    # Translation of image in homogenous coordinates (adding third coordinate)
    T = np.array([[1, 0, -x_min],
                 [0, 1, -y_min],
                 [0, 0, 1     ]])
    
    H_total = T @ H
    new_w = int(x_max - x_min)
    new_h = int(y_max - y_min)
    img_rotated = cv2.warpPerspective(img, H_total, (new_w, new_h), flags=cv2.INTER_CUBIC)

    return img_rotated

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Couldn't load file {path}")
    return img

# Testing code below
img = load_image("test1.jpg")
K = calibration.calib_mtx()
rotated_img = unrotate_image(img, [0,0,0,0,-0.1,-1], K)

# test image
cv2.imshow("img", rotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



    
    


