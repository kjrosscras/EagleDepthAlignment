import numpy as np
import laspy
import json
import cv2
import os
import pandas as pd
from scipy.spatial.transform import Rotation as R

# === CONFIGURATION ===
CAMERA_NAME = "front"
IMG_WIDTH = 4000
IMG_HEIGHT = 3000
IMAGE_TIMESTAMP = 1744400622.613818
RGB_IMAGE_NAME = f"{IMAGE_TIMESTAMP}.jpg"
LAS_PATH = "recalculate_result.las"
CALIB_PATH = "calib.json"
POSE_CSV = "recalculate_pose.csv"
DEPTH_OUT_PATH = f"depth_{CAMERA_NAME}_{IMAGE_TIMESTAMP}.png"

# === LOAD POINT CLOUD ===
las = laspy.read(LAS_PATH)
points = np.vstack((las.x, las.y, las.z)).T

# === LOAD CAMERA CALIBRATION ===
with open(CALIB_PATH) as f:
    calib = json.load(f)

K = np.array(calib["camera_info"][CAMERA_NAME]["K"]).reshape(3, 3)
D = np.array(calib["camera_info"][CAMERA_NAME]["coeff"])
T_camera_to_lidar = np.array(calib["out_put"][CAMERA_NAME]["transform_matrix"])

# === LOAD POSES ===
poses = pd.read_csv(POSE_CSV, header=None)
poses = poses.iloc[:, :9]
poses.columns = ['frame', 'timestamp', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']
poses['timestamp'] = poses['timestamp'].astype(float)

# === FIND CLOSEST POSE ===
closest_idx = (poses['timestamp'] - IMAGE_TIMESTAMP).abs().idxmin()
row = poses.loc[closest_idx]

# === EXTRACT POSE ===
pos_interp = np.array([row['x'], row['y'], row['z']])
rot_interp = R.from_quat([row['qx'], row['qy'], row['qz'], row['qw']])


# Move Up or Down
pos_interp[2] -= 0.5  # drop 10 cm vertically

# Move 10cm backward *relative to camera's actual forward*
cam_forward = rot_interp.apply([0, 0, 1])
pos_interp += cam_forward * 0.2  # flipped direction


# ✅ Optional: pitch correction (flip camera upright)
fix_rot = R.from_euler('x', 180, degrees=True)
rot_interp = fix_rot * rot_interp

# === BUILD TRANSFORM ===
T_world_to_pose = np.eye(4)
T_world_to_pose[:3, :3] = rot_interp.as_matrix()
T_world_to_pose[:3, 3] = pos_interp

T_world_to_lidar = T_camera_to_lidar @ T_world_to_pose

# === DEBUG AXES ===
print("\n--- Raw Camera Forward Vector (Z_cam) from pose ---")
print(rot_interp.apply([0, 0, 1]))

print("\n--- Camera Coordinate Axes ---")
rot = T_camera_to_lidar[:3, :3] @ T_world_to_pose[:3, :3]
print("X_cam (right):", rot @ np.array([1, 0, 0]))
print("Y_cam (down?):", rot @ np.array([0, 1, 0]))
print("Z_cam (forward?):", rot @ np.array([0, 0, 1]))

# === TRANSFORM AND PROJECT POINTS ===
points_h = np.hstack([points, np.ones((points.shape[0], 1))])
cam_points = (T_world_to_lidar @ points_h.T).T[:, :3]
valid = cam_points[:, 2] > 0
cam_points = cam_points[valid]

pixels = (K @ cam_points.T).T
pixels[:, 0] /= pixels[:, 2]
pixels[:, 1] /= pixels[:, 2]
z = cam_points[:, 2]

finite_mask = np.isfinite(pixels[:, 0]) & np.isfinite(pixels[:, 1]) & (z > 0)
pixels, z = pixels[finite_mask], z[finite_mask]
u = pixels[:, 0].astype(np.int32)
v = pixels[:, 1].astype(np.int32)

in_bounds = (u >= 0) & (u < IMG_WIDTH) & (v >= 0) & (v < IMG_HEIGHT)
u, v, z = u[in_bounds], v[in_bounds], z[in_bounds]

# === BUILD DEPTH MAP ===
depth_map = np.full((IMG_HEIGHT, IMG_WIDTH), np.inf, dtype=np.float32)
for x, y, depth in zip(u, v, z):
    if depth < depth_map[y, x]:
        depth_map[y, x] = depth

depth_map[depth_map == np.inf] = 0
depth_map_mm = (depth_map * 1000).astype(np.uint16)

# === SAVE OUTPUT ===
max_depth_mm = 8000
depth_normalized = np.clip(depth_map_mm, 0, max_depth_mm)
depth_normalized = (depth_normalized / max_depth_mm * 255).astype(np.uint8)
colored_depth = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)

# ✅ Flip vertically and horizontally if needed
colored_depth = cv2.flip(colored_depth, -1)
depth_map_mm = cv2.flip(depth_map_mm, -1)

cv2.imwrite(DEPTH_OUT_PATH, depth_map_mm)
cv2.imwrite(DEPTH_OUT_PATH.replace('.png', '_color.png'), colored_depth)

print(f"✅ Saved raw: {DEPTH_OUT_PATH}")
print(f"✅ Saved colored: {DEPTH_OUT_PATH.replace('.png', '_color.png')}")
