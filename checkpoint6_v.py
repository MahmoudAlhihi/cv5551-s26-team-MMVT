import cv2, numpy, time, torch
import open3d as o3d
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH

CUBE_SIZE = 0.025

robot_ip = ''

def get_transform_cube(observation, camera_intrinsic, camera_pose):
    """
    Calculate the transformation matrix for the cube relative to the robot base frame, 
    as well as relative to the camera frame.

    This function leverages text prompts to semantically segment a specific 
    cube (e.g., 'red cube') and determines the cube's pose using its 3D point cloud.

    Parameters
    ----------
    observation : list or tuple
        A collection containing [image, point_cloud], where image is the 
        RGB/BGRA array and point_cloud is the registered 3D point cloud.
    camera_intrinsic : numpy.ndarray
        The 3x3 intrinsic camera matrix.
    camera_pose : numpy.ndarray
        A 4x4 transformation matrix representing the camera's pose in the robot base frame (t_cam_robot).
        All translations are in meters.

    Returns
    -------
    tuple or None
        If successful, returns a tuple (t_robot_cube, t_cam_cube) where both 
        are 4x4 transformation matrices with translations in meters. 
        If no matching object is segmented, returns None.
    """
    image, point_cloud = observation
    # TODO

    # Using this: https://www.open3d.org/docs/latest/python_api/open3d.geometry.OrientedBoundingBox.html#open3d.geometry.OrientedBoundingBox.create_from_points

    """
    Estimating the cube pose from the raw ZED point cloud.

    Steps:
    1. Flatten the HxWx3 point cloud into an Nx3 list of points.
    2. Remove invalid points (NaN / inf).
    3. Convert the points into an Open3D PointCloud.
    4. Remove the dominant plane (the tabletop).
    5. Cluster the remaining points into separate objects.
    6. Pick the cluster whose bounding box dimensions best match CUBE_SIZE.
    7. Fit an oriented bounding box to that cluster.
    8. Convert that pose from camera frame to robot frame.
    """

     # Flatten HxWx3 point cloud image into an Nx3 array of XYZ points.
    pts = point_cloud[..., :3].reshape(-1, 3)

    # Remove invalid 3D points.
    pts = pts[numpy.isfinite(pts).all(axis=1)]

    if len(pts) < 50:
        print("Not enough valid points in point cloud after filtering.")
        return None

    # Create Open3D point cloud from valid points.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Remove the dominant plane, which should usually be the tabletop.
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.005,
        ransac_n=3,
        num_iterations=1000
    )

    # Keep only points that are NOT on the table plane.
    object_pcd = pcd.select_by_index(inliers, invert=True)

    if len(object_pcd.points) < 20:
        print("Not enough non-plane points remain after removing table.")
        return None

    # Cluster the remaining points into separate objects.
    # This helps separate the cube from the cup or any other clutter.
    labels = numpy.array(
        object_pcd.cluster_dbscan(
            eps=0.02,         # neighborhood radius in meters
            min_points=30,    # minimum cluster size
            print_progress=False
        )
    )

    if labels.size == 0 or labels.max() < 0:
        print("No object clusters found.")
        return None

    best_cluster = None
    best_score = float("inf")

    # Examine each cluster and choose the one most cube-like.
    for label in range(labels.max() + 1):
        idx = numpy.where(labels == label)[0]
        if len(idx) < 20:
            continue

        cluster = object_pcd.select_by_index(idx)

        # Use an axis-aligned box first just to score the size.
        aabb = cluster.get_axis_aligned_bounding_box()
        extent = numpy.array(aabb.get_extent())

        # Score how close this object's size is to the real cube size.
        # Smaller score = more likely to be the cube.
        score = numpy.sum((extent - CUBE_SIZE) ** 2)

        print(f"Cluster {label}: extent = {extent}, score = {score}")

        if score < best_score:
            best_score = score
            best_cluster = cluster

    if best_cluster is None:
        print("Could not find a cube-sized cluster.")
        return None

    # Fit an oriented bounding box to the chosen cube cluster.
    obb = best_cluster.get_oriented_bounding_box()

    print("Chosen OBB center:", obb.center)
    print("Chosen OBB extent:", obb.extent)

    # Build cube pose in camera frame.
    t_cam_cube = numpy.eye(4)
    t_cam_cube[:3, :3] = obb.R
    t_cam_cube[:3, 3] = obb.center

    # Convert cube pose from camera frame to robot frame.
    t_robot_cube = numpy.linalg.inv(camera_pose) @ t_cam_cube
    # OBB center is the geometric center of the cube.
    # checkpoint1 grasp/place behavior works better when the pose translation
    # corresponds to the cube's top face rather than its center.
    # Since the cube is 0.025 m tall, shift by half the cube height.
    t_robot_cube[2, 3] -= CUBE_SIZE / 2.0

    return t_robot_cube, t_cam_cube

def main():

    # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    # Initialize Lite6 Robot
    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)
    time.sleep(0.5)

    try:
        # Get Observation
        cv_image = zed.image
        point_cloud = zed.point_cloud

       # First estimate camera pose relative to robot using checkpoint 0.
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            print("Failed to compute camera-to-robot transform.")
            return

        # Then estimate cube pose from the point cloud.
        result = get_transform_cube([cv_image, point_cloud], camera_intrinsic, t_cam_robot)
        if result is None:
            print("Cube not found.")
            return

        t_robot_cube, t_cam_cube = result

            
        # TODO
        
        # Visualization
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube)
        cv2.namedWindow('Verifying Cube Pose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Cube Pose', 1280, 720)
        cv2.imshow('Verifying Cube Pose', cv_image)
        key = cv2.waitKey(0)
        
        if key == ord('k'):
        
            cv2.destroyAllWindows()
        
            # TODO

            grasp_cube(arm, t_robot_cube)
            place_cube(arm, t_robot_cube)
    
    finally:
        # Close Lite6 Robot
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()
    
        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()
