from checkpoint3 import CubePoseDetector

import cv2, time
from xarm.wrapper import XArmAPI
import numpy

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH

# TODO
STACK_HEIGHT = 0.025   # Determine a suitable height yourself

robot_ip = '192.168.1.182'

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

        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            return
        cube_pose_detector.camera_pose = t_cam_robot

        red = cube_pose_detector.get_transforms(cv_image, 'red cube')
        if red == None:
            return 
        green = cube_pose_detector.get_transforms(cv_image, 'green cube')
        if green == None:
            return 
        t_robot_red, t_cam_red = red
        t_robot_green, t_cam_green = green

        draw_pose_axes(cv_image, camera_intrinsic, t_cam_red)
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_green)
        cv2.namedWindow("Verifying cube poses", cv2.WINDOW_NORMAL)
        cv2.imshow("Verifying cube poses", cv_image)
        key = cv2.waitKey(0)

        if key == ord ('k'):
            cv2.destroyAllWindows()
            
            grasp_cube(arm, t_robot_red)
            green_copy = numpy.copy(t_robot_green)
            green_copy[2,3] -= STACK_HEIGHT

            place_cube(arm, green_copy)

    
    finally:
        # Close Lite6 Robot
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()
