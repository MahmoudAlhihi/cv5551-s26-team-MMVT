import cv2, numpy, time
import open3d as o3d
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH
from checkpoint6 import CUBE_SIZE

cube_prompt = 'blue cube'
robot_ip = '192.168.1.182'

class CubePoseDetector:
    """
    A detector to robustly identify and locate a specific cube in the scene.

    This class leverages text prompts to semantically segment a specific cube (e.g., 
    'blue cube') and determine the cube's pose by its 3D point cloud.
    """

    def __init__(self, camera_intrinsic):
        """
        Initialize the CubePoseDetector with camera parameters.

        Parameters
        ----------
        camera_intrinsic : numpy.ndarray
            The 3x3 intrinsic camera matrix.
        """
        self.camera_intrinsic = camera_intrinsic
        # TODO
        self.camera_pose = None  # set to t_cam_robot before calling get_transforms
        self.color_hsv_ranges = {
            'red':   [(numpy.array([0,   80,  50]), numpy.array([15,  255, 255])),
                      (numpy.array([165, 80,  50]), numpy.array([180, 255, 255]))],
            'green': [(numpy.array([35,  50,  50]), numpy.array([80,  255, 255]))],
            'blue':  [(numpy.array([90,  80,  60]), numpy.array([130, 255, 255]))],
        }


    def get_transforms(self, observation, cube_prompt):
        """
        Calculate the transformation matrix for a specific prompted cube relative to the robot base frame,
        as well as relative to the camera frame.

        Parameters
        ----------
        observation : list or tuple
            A collection containing [image, point_cloud], where image is the 
            RGB/BGRA array and point_cloud is the registered 3D point cloud.
        cube_prompt : str
            The text prompt used to segment the target object (e.g., 'blue cube').

        Returns
        -------
        tuple or None
            If successful, returns a tuple (t_robot_cube, t_cam_cube) where both 
            are 4x4 transformation matrices with translations in meters. 
            If no matching object is segmented, returns None.
        """
        image, point_cloud = observation

        # TODO
        # parse colour from prompt
        color = None
        for c in self.color_hsv_ranges:
            if c in cube_prompt.lower():
                color = c
                break
        if color is None:
            print(f'Could not parse colour from prompt: "{cube_prompt}"')
            return None
 
        # build HSV mask
        if len(image.shape) == 3 and image.shape[2] == 4:
            bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            bgr = image.copy()
 
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = numpy.zeros(hsv.shape[:2], dtype=numpy.uint8)
        for lo, hi in self.color_hsv_ranges[color]:
            mask |= cv2.inRange(hsv, lo, hi)
 
        kernel = numpy.ones((3, 3), numpy.uint8)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   kernel, iterations=2)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
 
        if mask.sum() == 0:
            print(f'No pixels matched colour "{color}".')
            return None
 
        # extract 3D points under the mask and filter NaNs
        raw_points = point_cloud[mask > 0][:, :3]
        cpoints    = raw_points[numpy.all(numpy.isfinite(raw_points), axis=1)]
 
        if len(cpoints) < 50:
            print(f'Too few valid points ({len(cpoints)}) for "{color}" cube')
            return None
 
        # Open3D oriented bounding box for pose
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cpoints.astype(numpy.float64))
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
 
        if len(pcd.points) < 4:
            print('Too less inlier points after outlier removal')
            return None
 
        obb = pcd.get_oriented_bounding_box()
 
        # build camera-frame transform
        t_cam_cube          = numpy.eye(4)
        t_cam_cube[:3, :3]  = numpy.asarray(obb.R)
        t_cam_cube[:3,  3]  = numpy.asarray(obb.center)
 
        # transforn to robot base frame - camera_pose is t_cam_robot
        t_robot_cube = numpy.linalg.inv(self.camera_pose) @ t_cam_cube
 
        return t_robot_cube, t_cam_cube



def main():

    # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    # Initialize Cube Pose Detector
    cube_pose_detector = CubePoseDetector(camera_intrinsic)

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
            print('Failed to compute cam-robot transf')
            return
 
        cube_pose_detector.camera_pose = t_cam_robot
 
        result = cube_pose_detector.get_transforms([cv_image, point_cloud], cube_prompt)
        if result is None:
            print(f'Could not detect "{cube_prompt}"')
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
