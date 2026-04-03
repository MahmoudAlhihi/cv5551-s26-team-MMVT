import cv2
import numpy
import open3d as o3d
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot

ROBOT_IP = '192.168.1.182'

# Table bounds in robot frame (metres) — adjust to your workspace
X_MIN, X_MAX =  0.05,  0.50
Y_MIN, Y_MAX = -0.35,  0.35
Z_MIN, Z_MAX = -0.01,  0.50

VOXEL_SIZE = 0.01  # 10 mm instead of 5


def voxelize_table(point_cloud, T_cam_robot):
    pts = point_cloud[:, :, :3].reshape(-1, 3)
    pts = pts[numpy.all(numpy.isfinite(pts), axis=1)]

    # Downsample before transform — keep every 4th point
    pts = pts[::4]

    pts_m = pts / 1000.0
    ones = numpy.ones((len(pts_m), 1))
    T_cam_robot_inv = numpy.linalg.inv(T_cam_robot)
    pts_robot = (T_cam_robot_inv @ numpy.hstack([pts_m, ones]).T).T[:, :3]

    mask = (
        (pts_robot[:, 0] > X_MIN) & (pts_robot[:, 0] < X_MAX) &
        (pts_robot[:, 1] > Y_MIN) & (pts_robot[:, 1] < Y_MAX) &
        (pts_robot[:, 2] > Z_MIN) & (pts_robot[:, 2] < Z_MAX)
    )
    table_pts = pts_robot[mask]
    print(f"Total points: {len(pts_robot)}, after crop: {len(table_pts)}")

    if len(table_pts) < 100:
        print("Not enough points in table region.")
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(table_pts)

    # Further downsample with Open3D's voxel filter
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f"Points after downsampling: {len(pcd.points)}")

    z_vals = numpy.asarray(pcd.points)[:, 2]
    z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-8)
    colors = numpy.zeros((len(pcd.points), 3))
    colors[:, 0] = z_norm
    colors[:, 2] = 1.0 - z_norm
    pcd.colors = o3d.utility.Vector3dVector(colors)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd, voxel_size=VOXEL_SIZE
    )

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0]
    )

    print(f"Voxel grid created: voxel_size={VOXEL_SIZE*1000:.0f} mm")
    o3d.visualization.draw_geometries(
        [voxel_grid, axes],
        window_name="Table Voxel Grid (Robot Frame)",
        width=1280, height=720
    )

    return voxel_grid

def main():
    zed = ZedCamera()

    try:
        cv_image, point_cloud = zed.get_synchronized_frame()

        T_cam_robot = get_transform_camera_robot(cv_image, zed.camera_intrinsic)
        if T_cam_robot is None:
            print("Could not compute camera-robot transform.")
            return

        print(f"Point cloud shape: {point_cloud.shape}")
        voxel_grid = voxelize_table(point_cloud, T_cam_robot)

    finally:
        zed.close()


if __name__ == "__main__":
    main()