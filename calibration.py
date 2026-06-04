import numpy as np
import cv2 as cv
import glob

def calib_mtx():
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    square_size = 25.0 # mm

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0), ... ,(6,5,0)
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
    objp[:,:2] *= square_size

    # Arrays to store object points and image points from all the images.
    imgpoints = [] # 2d points in image plane.
    objpoints = [] # 3d point in real world space

    images = glob.glob('chessboard2.jpg')

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (8,6), None)
        print(ret)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

    ret, K, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return K

