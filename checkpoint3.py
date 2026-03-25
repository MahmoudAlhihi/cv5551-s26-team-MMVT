import cv2, numpy, time
from pupil_apriltags import Detector
from xarm.wrapper import XArmAPI
from lang_sam import LangSAM
from PIL import Image

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH, CUBE_TAG_FAMILY, CUBE_TAG_ID, CUBE_TAG_SIZE

cube_prompt = 'blue cube'
robot_ip = '192.168.1.158'

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
        self.model = LangSAM()
        self.detector = Detector(families=CUBE_TAG_FAMILY)
        self.camera_pose = None
        

    def get_transforms(self, observation, cube_prompt):
        """
        Calculate the transformation matrix for a specific prompted cube relative to the robot base frame,
        as well as relative to the camera frame.

        Parameters
        ----------
        observation : numpy.ndarray
            The input image from the camera. Can be a color (BGRA/BGR) or grayscale image.
        cube_prompt : str
            The text prompt used to segment the target object (e.g., 'blue cube').

        Returns
        -------
        tuple or None
            If successful, returns a tuple (t_robot_cube, t_cam_cube) where both 
            are 4x4 transformation matrices with translations in meters. 
            If no matching object or tag is found, returns None.
        """
        if len(observation.shape) > 2:
            if observation.shape[2] == 4:
                gray = cv2.cvtColor(observation, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        else:
            gray = observation

        fx = self.camera_intrinsic[0,0]
        fy = self.camera_intrinsic[1,1]
        cx = self.camera_intrinsic[0,2]
        cy = self.camera_intrinsic[1,2]
        
        if len (observation.shape) > 2:
            if observation.shape[2] == 4:
                new_image = cv2.cvtColor(observation, cv2.COLOR_BGRA2RGB)
            else:
                new_image = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
        
        new_observation = Image.fromarray(new_image)

        tags = self.detector.detect(gray, estimate_tag_pose=True, camera_params = (fx,fy,cx,cy), tag_size=CUBE_TAG_SIZE)
        masks, boxes, phrases, logits = self.model.predict(new_observation, cube_prompt)

        #check if model found anything 
        best_tag = None
        best_dist = float('inf')
        if len(masks) == 0:
            return None
        for tag in tags:
            center_x = tag.center[0] # (column)
            center_y = tag.center[1] # (row)
            if masks[0][int(center_y)][int(center_x)]:
                dist = numpy.linalg.norm(tag.pose_t)
                if dist < best_dist:
                    best_dist = dist
                    best_tag = tag
                
        if best_tag is None:
                return None
        
        t_cam_cube = numpy.eye(4)
        t_cam_cube[:3, :3] = best_tag.pose_R
        t_cam_cube[:3, 3] = best_tag.pose_t.flatten()

        t_robot_cube = numpy.linalg.inv(self.camera_pose) @ t_cam_cube
        return (t_robot_cube, t_cam_cube)
        

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

        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            return
        
        cube_pose_detector.camera_pose = t_cam_robot
        result = cube_pose_detector.get_transforms(cv_image, cube_prompt)
        if result is None:
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
