import cv2, numpy, time
from pupil_apriltags import Detector
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH, CUBE_TAG_FAMILY, CUBE_TAG_ID, CUBE_TAG_SIZE

cube_prompt = 'blue cube'
robot_ip = '192.168.1.182'

class CubePoseDetector:
    """
    A detector to robustly identify and locate a specific cube in the scene.

    This class leverages text prompts to semantically segment a specific cube (e.g., 
    'blue cube') and determine the cube's pose by the AprilTags.
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
        self.detector = Detector(families=CUBE_TAG_FAMILY)
        self.camera_pose = None  # set before calling get_transforms
        self.color_hsv_ranges = {
            'red':   [(numpy.array([0,100,60]), numpy.array([10,255,255])),
        (numpy.array([170,100,60]), numpy.array([180,255,255]))],
            'green': [(numpy.array([45,60,60]), numpy.array([75,255,255]))],
            'blue':  [(numpy.array([90,80,60]), numpy.array([130,255,255]))],
        }

    def get_transforms(self, observation, cube_prompt):
        """
        Calculate the transformation matrix for a specific prompted cube relative to the robot base frame,
        as well as relative to the camera frame.

        Parameters
        ----------
        observation : numpy.ndarray
            The input image from the camera. Can be a colour (BGRA/BGR) or grayscale image.
        cube_prompt : str
            The text prompt used to segment the target object (e.g., 'blue cube').

        Returns
        -------
        tuple or None
            If successful, returns a tuple (t_robot_cube, t_cam_cube) where both 
            are 4x4 transformation matrices with translations in meters. 
            If no matching object or tag is found, returns None.
        """
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
 
        # prep imgs
        if observation.shape[2] == 4:
            bgr  = cv2.cvtColor(observation, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(observation, cv2.COLOR_BGRA2GRAY)
        else:
            bgr  = observation
            gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
 
        hsv    = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        print(hsv)
        h, w   = gray.shape
 
        # detect AprilTags with pose est
        fx = self.camera_intrinsic[0, 0]
        fy = self.camera_intrinsic[1, 1]
        cx = self.camera_intrinsic[0, 2]
        cy = self.camera_intrinsic[1, 2]
        tags = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=CUBE_TAG_SIZE
        )

        print(f"Detected {len(tags)} tags, looking for {cube_prompt}")
 
        # For each detected tag, sample HSV values just outside the tag boundary
        # to determine the cube's color. The tag corners are ordered:
        # bottom-left, bottom-right, top-right, top-left.
        # We compute the midpoint of each edge and move slightly outward from
        # the tag center so we land on the colored cube face instead of the
        # black/white AprilTag itself

        # for each tag:
        # 1 sample multiple HSV points around the tag
        # 2 compute the median HSV for robustness to noise
        # 3 check if the color matches the requested cube (e.g., "blue cube")
        # 4 among all matching candidates, select the closest cube (smallest depth)
        if len(tags) == 0:
            print("No tags detected")
            return None

        # eval each tag
        best_tag = None
        best_dist = float('inf')

        for tag in tags:
            corners  = tag.corners
            centre   = tag.center 
 
            # sample hsv values just outside tag edges
            samples = []
            for i in range(4):
                mid = (corners[i] + corners[(i + 1) % 4]) / 2.0
                # push midpoint outward from tag centre by 1.5x tag half-width
                direction  = mid - centre
                pt  = centre + 1.5 * direction
                px = int(numpy.clip(pt[0], 0, w - 1))
                py = int(numpy.clip(pt[1], 0, h - 1))

                samples.append(hsv[py, px])
 
            if len(samples) == 0:
                continue

            # use median hsv for robustness to noise
            median_hsv = numpy.median(numpy.array(samples), axis=0)
 
            # colour matching
            match = False
            for lo, hi in self.color_hsv_ranges[color]:
                if numpy.all(median_hsv >= lo) and numpy.all(median_hsv <= hi):
                    match = True
                    break

            if not match:
                continue
 
            # choose closest cube
            dist = numpy.linalg.norm(tag.pose_t)

            if dist < best_dist:
                best_dist = dist
                best_tag = tag

        if best_tag is None:
            print(f'No matching "{cube_prompt}" found.')
            return None
 
        # build transforms
        t_cam_cube = numpy.eye(4)
        t_cam_cube[:3, :3] = best_tag.pose_R
        t_cam_cube[:3, 3]  = best_tag.pose_t.flatten()
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

        t_cam_cube = None
        # TODO
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            print('Failed to compute cam robot transform')
            return
 
        cube_pose_detector.camera_pose = t_cam_robot
 
        result = cube_pose_detector.get_transforms(cv_image, cube_prompt)
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

