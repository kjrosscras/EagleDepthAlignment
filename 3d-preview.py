import numpy as np
import open3d as o3d
import laspy
import pandas as pd
from scipy.spatial.transform import Rotation as R

# === CONFIGURATION ===
LAS_PATH = "recalculate_result.las"
POSE_CSV = "recalculate_pose.csv"
TIMESTAMP = 1744400622.613818

DOWNSAMPLE_RATIO = 0.2  # Simulate transparency (keep 20%)
POINT_SIZE = 1.0        # Smaller = more transparent effect
BG_COLOR = [1, 1, 1]    # White background

# === LOAD POINT CLOUD ===
las = laspy.read(LAS_PATH)
points = np.vstack((las.x, las.y, las.z)).T
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.paint_uniform_color([0.6, 0.6, 0.6])

# ✅ Downsample to make it "see-through"
pcd = pcd.random_down_sample(DOWNSAMPLE_RATIO)

# === LOAD POSE ===
poses = pd.read_csv(POSE_CSV, header=None)
poses.columns = ['frame', 'timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
poses['timestamp'] = poses['timestamp'].astype(float)
row = poses.loc[(poses['timestamp'] - TIMESTAMP).abs().idxmin()]

pos = np.array([row['x'], row['y'], row['z']])
rot = R.from_quat([row['qx'], row['qy'], row['qz'], row['qw']])


# === COORDINATE AXES ===
def create_axes(origin, rotation, size=0.3):
    axes = []
    directions = {
        'x': rotation.apply([1, 0, 0]),  # red
        'y': rotation.apply([0, 1, 0]),  # green
        'z': rotation.apply([0, 0, 1]),  # blue
    }
    colors = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
    for axis, vec in directions.items():
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([origin, origin + size * vec])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([colors[axis]])
        axes.append(line)
    return axes

# === VISUALIZE ===
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Camera Pose Preview', width=1600, height=1200)
vis.add_geometry(pcd)
for axis in create_axes(pos, rot):
    vis.add_geometry(axis)

opt = vis.get_render_option()
opt.background_color = np.array(BG_COLOR)
opt.point_size = POINT_SIZE
opt.show_coordinate_frame = False

vis.run()
vis.destroy_window()
