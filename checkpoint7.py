from checkpoint6 import get_transform_cube

import cv2, numpy, time
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, GRIPPER_LENGTH
from checkpoint2 import place_in_basket, BASKET_POSE

robot_ip = '192.168.1.155'

def main():

    zed = ZedCamera()
    camera_intristnic = zed.camera_intrinsic

    # Initialize ZED Camera
    cv_image, point_cloud = zed.get_synchronized_frame()

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
        # Compute camera → robot transform
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intristnic)
        if t_cam_robot is None:
            print("Failed to detect registration tags")
            return

        result = get_transform_cube([cv_image, point_cloud], camera_intristnic, t_cam_robot)
        if result is None:
            print("Cube detection failed")
            return

        t_robot_cube, t_cam_cube = result
        
        
        # Visualization
        draw_pose_axes(cv_image, camera_intristnic, t_cam_cube)
        cv2.imwrite('verify_pose_cp7.jpg', cv_image)
        print("Saved verify_pose_cp7.jpg — check it, then type 'k' + Enter to execute:")
        if input().strip() != 'k':
            print("Aborted")
            return

            # TODO
            # pick and place into basket
            grasp_cube(arm, t_robot_cube)
            time.sleep(0.5)
            place_in_basket(arm, BASKET_POSE)
            
    
    finally:
        # Close Lite6 Robot
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()