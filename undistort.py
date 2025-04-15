import cv2
import numpy as np
import json

# === LOAD CALIBRATION ===
with open("calib.json") as f:
    calib = json.load(f)

K = np.array(calib["camera_info"]["front"]["K"]).reshape(3, 3)
D = np.array(calib["camera_info"]["front"]["coeff"])[:4]  # fisheye uses 4 params max

# === LOAD IMAGE ===
img = cv2.imread("1744400622.613818.jpg")
h, w = img.shape[:2]

# === UNDISTORT ===
undistorted = cv2.fisheye.undistortImage(img, K, D, Knew=K, new_size=(w, h))

cv2.imwrite("undistorted_fisheye_test.jpg", undistorted)
