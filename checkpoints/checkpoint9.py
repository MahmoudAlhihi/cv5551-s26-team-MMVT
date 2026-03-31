from checkpoints.checkpoint8 import CubePoseDetector

import cv2, time
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoints.checkpoint0 import get_transform_camera_robot
from checkpoints.checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH
from checkpoints.checkpoint4 import STACK_HEIGHT

robot_ip = '192.168.1.155'

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
        cv_image, point_cloud = zed.get_synchronized_frame()

        # TODO
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            print('Failed to compute cam-robot transform')
            return
 
        cube_pose_detector.camera_pose = t_cam_robot
 
        # detect red and green cubes
        redr = cube_pose_detector.get_transforms([cv_image, point_cloud], 'red cube')
        greenr = cube_pose_detector.get_transforms([cv_image, point_cloud], 'green cube')
 
        if redr is None or greenr is None:
            print('Could not locate both cubes for stacking')
            return
 
        t_robot_red, t_cam_red = redr
        t_robot_green, t_cam_green = greenr
 
        # visualization
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_red)
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_green)
        cv2.imshow('Checkpoint 9', cv_image)
        print("Press 'k' to execute stacking")
        if cv2.waitKey(0) != ord('k'):
            return
        cv2.destroyAllWindows()
 
        # pick up the red cube
        grasp_cube(arm, t_robot_red)
        time.sleep(0.5)
 
        # compute stack pose: on top of green cube
        t_robot_stack = t_robot_green.copy()
        t_robot_stack[2, 3] -= STACK_HEIGHT
        t_robot_stack[:3, :3] = t_robot_green[:3, :3]
 
        # place red cube on green cube
        place_cube(arm, t_robot_stack)
    
    finally:
        # Close Lite6 Robot
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()
