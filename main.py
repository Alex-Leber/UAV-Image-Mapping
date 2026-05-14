import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Couldn't load file {path}")
    return img

# tilt_strength in range [0,1]
def apply_perspective(img, tilt_strength, resize):
    img = cv2.resize(img, None, fx=resize, fy=resize)
    h, w = img.shape[:2]
    
    src_corners = np.float32([[0,0], [0,h], [w,h], [w,0]])
    delta = w * tilt_strength
    dst_corners = np.float32([[delta,0], [0,h], [w, h], [w-delta,0]])
    M = cv2.getPerspectiveTransform(src_corners, dst_corners)
    warped = cv2.warpPerspective(img, M, (w,h))
    return warped, M

img = load_image("testPhoto.jpeg")
warped, _ = apply_perspective(img, 0.3, 0.3)

cv2.imshow("warped image", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()




# Test homography with planar images (not ideal for rough terrain)
# def homography_transformation(img1, img2):
#     img1 = cv2.imread(img1)
#     img2 = cv2.imread(img2)

#     img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#     ret1, corners1 = cv2.findChessboardCorners(img1, (9,6))
#     ret2, corners2 = cv2.findChessboardCorners(img2, (9,6))

#     H, _ = cv2.findHomography(corners1, corners2)
#     print(H)

#     img1_warp = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))

#     return img1_warp

# cv2.imshow("image", homography_transformation("img1.jpg", "img2.jpg"), )
# cv2.waitKey(0)
# cv2.destroyAllWindows
