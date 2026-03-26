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
robot_ip = ''

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
        self.camera_pose = None  # set from main() as t_cam_robot

        # HSV ranges for the three cube colors
        self.color_hsv_ranges = {
            'red': [
                (numpy.array([0, 80, 50]),   numpy.array([15, 255, 255])),
                (numpy.array([165, 80, 50]), numpy.array([180, 255, 255]))
            ],
            'green': [
                (numpy.array([35, 50, 50]),  numpy.array([85, 255, 255]))
            ],
            'blue': [
                (numpy.array([90, 80, 50]),  numpy.array([135, 255, 255]))
            ],
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

        """
        Returns (t_robot_cube, t_cam_cube) for the prompted cube, or None.
        """
        image, point_cloud = observation

        if self.camera_pose is None:
            print("camera_pose was not set before calling get_transforms.")
            return None

        # Parse requested color from prompt like "blue cube"
        color = None
        prompt_lower = cube_prompt.lower()
        for c in self.color_hsv_ranges:
            if c in prompt_lower:
                color = c
                break

        if color is None:
            print(f'Could not parse color from prompt: "{cube_prompt}"')
            return None

        # Convert image to BGR if needed
        if len(image.shape) == 3 and image.shape[2] == 4:
            bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            bgr = image.copy()

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Build color mask
        mask = numpy.zeros(hsv.shape[:2], dtype=numpy.uint8)
        for lo, hi in self.color_hsv_ranges[color]:
            mask |= cv2.inRange(hsv, lo, hi)

        # Clean mask a little (learned about kernal morphology in CSCI3081 and CSCI4551 [basically helps clean up shapes in greyscale images and reduce noise!])
        kernel = numpy.ones((5, 5), numpy.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Keep only the largest connected component so we isolate one cube
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if num_labels <= 1:
            print(f'No visible region found for "{cube_prompt}".')
            return None

        largest_label = 1 + numpy.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(numpy.uint8) * 255

        # Use the 2D mask to select only that cube's 3D points
        pts = point_cloud[..., :3][mask > 0]

        # Remove invalid 3D points
        pts = pts[numpy.isfinite(pts).all(axis=1)]

        if len(pts) < 30:
            print(f'Not enough valid 3D points for "{cube_prompt}".')
            return None

        # Build Open3D point cloud from masked points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # Optional light downsampling for stability
        pcd = pcd.voxel_down_sample(voxel_size=0.002)

        if len(pcd.points) < 20:
            print("Too few points remain after downsampling.")
            return None

        # Fit oriented bounding box to the selected cube points
        obb = pcd.get_oriented_bounding_box()

        print(f'{color} OBB center:', obb.center)
        print(f'{color} OBB extent:', obb.extent)

        # Build pose in camera frame
        t_cam_cube = numpy.eye(4)
        t_cam_cube[:3, :3] = obb.R
        t_cam_cube[:3, 3] = obb.center

        # Convert to robot frame
        t_robot_cube = numpy.linalg.inv(self.camera_pose) @ t_cam_cube

        # Adjust from cube center to cube top face for checkpoint1 grasping logic
        t_robot_cube[2, 3] -= CUBE_SIZE / 2.0

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

        # Visualization
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube)
        cv2.namedWindow('Verifying Cube Pose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Cube Pose', 1280, 720)
        cv2.imshow('Verifying Cube Pose', cv_image)
        key = cv2.waitKey(0)
    
        if key == ord('k'):
            cv2.destroyAllWindows()

            # TODO
    
    finally:
        # Close Lite6 Robot
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()
