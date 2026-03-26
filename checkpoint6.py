import cv2, numpy, time, torch
import open3d as o3d
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH

CUBE_SIZE = 0.025

robot_ip = '192.168.1.166'

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
    # semantic segmentation (red Mask)
    # red spans the 0 and 180 boundaries in hsv
    if image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr = image
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, numpy.array([0, 80, 80]), numpy.array([10, 255, 255]))
    m2 = cv2.inRange(hsv, numpy.array([160, 80, 80]), numpy.array([180, 255, 255]))
    mask = cv2.bitwise_or(m1,m2)
    
    kernel= numpy.ones((5,5), numpy.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('Red Mask', mask)
    cv2.waitKey(0)

    # filtering NaNs and extracting points
    # point_cloud is (H, W, 4) -> [x, y, z, color]
    raw_points = point_cloud[mask > 0][:, :3]
    
    # removing invalid depth readings (NaN or Inf)
    cpoints = raw_points[numpy.all(numpy.isfinite(raw_points), axis=1)]

    if len(cpoints) < 50:
        print("Not enough valid points detected")
        return None
    print(f"Point cloud sample: {point_cloud[600, 1100, :3]}")

    # using Open3D for Oriented Bounding Box
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cpoints)
    
    # get center and rotation from the geometry
    obb = pcd.get_oriented_bounding_box()
    centroid = numpy.mean(cpoints, axis=0)/1000.0 # mm to meters
    
    print(f"Centroid: {centroid}")
    print(f"Mask pixel count: {numpy.sum(mask > 0)}")
    print(f"Centroid (m): {centroid}")
    print(f"Number of valid 3D points: {len(cpoints)}")
    
    rotation_matrix = obb.R
    if numpy.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:,2] *= -1

    # constructing cam to cube transf
    t_cam_cube = numpy.eye(4)
    t_cam_cube[:3, :3] = rotation_matrix
    t_cam_cube[:3, 3] = centroid

    # constructing robot to cube transf
    # camera_pose is t_robot_cam 
    t_robot_cube = numpy.linalg.inv(camera_pose) @ t_cam_cube

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

        t_cam_cube = None
        # TODO
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            print("Failed to detect registration tags")
            return

        # detecting cube using pure vis
        result = get_transform_cube([cv_image, point_cloud], camera_intrinsic, t_cam_robot)
        
        if result is None:
            print("Cube detection failed")
            return
            
        t_robot_cube, t_cam_cube = result
        
        # Visualization
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube)
        cv2.namedWindow('Verifying Cube Pose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Cube Pose', 1280, 720)
        cv2.imshow('Verifying Cube Pose', cv_image)
        key = cv2.waitKey(0)

        if key == ord('k'):
            cv2.destroyAllWindows()

            # TODO
            # grasp
            grasp_cube(arm, t_robot_cube)
            time.sleep(0.5)
            # place
            place_cube(arm, t_robot_cube)
            # return home
            arm.move_gohome(wait=True)
    
    finally:
        # Close Lite6 Robot
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()
