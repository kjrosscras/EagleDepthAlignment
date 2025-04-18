import numpy as np
import laspy
import json
import cv2
import os
import pandas as pd
from scipy.spatial.transform import Rotation as R

# === CONFIGURATION ===
CAMERA_NAME = "front"
IMG_WIDTH = 1000
IMG_HEIGHT = 750
IMG_FOLDER = "undistorted"
OUT_FOLDER = "depth_outputs_lowres"
os.makedirs(OUT_FOLDER, exist_ok=True)

LAS_PATH = "recalculate_result.las"
CALIB_PATH = "calib.json"
POSE_CSV = "trj_imu_freq.txt"

# === LOAD POINT CLOUD ===
las = laspy.read(LAS_PATH)
points = np.vstack((las.x, las.y, las.z)).T

# === LOAD CALIBRATION ===
with open(CALIB_PATH) as f:
    calib = json.load(f)

K_full = np.array(calib["camera_info"][CAMERA_NAME]["K"]).reshape(3, 3)

# Scale intrinsics for lower resolution
scale_x = IMG_WIDTH / 4040
scale_y = IMG_HEIGHT / 2876
K = K_full.copy()
K[0, 0] *= scale_x  # fx
K[1, 1] *= scale_y  # fy
K[0, 2] *= scale_x  # cx
K[1, 2] *= scale_y  # cy

T_camera_to_lidar = np.array(calib["out_put"][CAMERA_NAME]["transform_matrix"])

# === LOAD POSES ===
poses = pd.read_csv(POSE_CSV, header=None, delim_whitespace=True)
poses.columns = ['timestamp', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']
poses['timestamp'] = poses['timestamp'].astype(float)

# === LOOP OVER IMAGES ===
for filename in sorted(os.listdir(IMG_FOLDER)):
    if not filename.endswith(".jpg"):
        continue

    try:
        timestamp = float(os.path.splitext(filename)[0])
    except ValueError:
        print(f"⚠️ Skipping {filename} — invalid timestamp format.")
        continue

    # === FIND CLOSEST POSE ===
    closest_idx = (poses['timestamp'] - timestamp).abs().idxmin()
    row = poses.loc[closest_idx]

    # === EXTRACT POSE ===
    pos_interp = np.array([-row['x'], row['y'], row['z']])
    rot_interp = R.from_quat([-row['qw'], -row['qx'], -row['qy'], row['qz']])



    # === BUILD TRANSFORM ===
    T_world_to_pose = np.eye(4)
    T_world_to_pose[:3, :3] = rot_interp.as_matrix()
    T_world_to_pose[:3, 3] = pos_interp

    T_world_to_lidar = T_camera_to_lidar @ T_world_to_pose



    # === TRANSFORM POINTS ===
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

    # === SAVE ONLY COLORIZED DEPTH ===
    max_depth_mm = 8000
    depth_normalized = np.clip(depth_map_mm, 0, max_depth_mm)
    depth_normalized = (depth_normalized / max_depth_mm * 255).astype(np.uint8)
    colored_depth = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)

    name = os.path.splitext(filename)[0]
    color_path = os.path.join(OUT_FOLDER, f"depth_{CAMERA_NAME}_{name}_color.png")
    cv2.imwrite(color_path, colored_depth)

    print(f"✅ Saved: {color_path}")
