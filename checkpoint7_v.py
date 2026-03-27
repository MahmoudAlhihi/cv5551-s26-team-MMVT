from checkpoint6 import get_transform_cube

import cv2, numpy, time
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, GRIPPER_LENGTH
from checkpoint2 import place_in_basket, BASKET_POSE

robot_ip = ''

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
        
        # Compute camera pose relative to robot base using checkpoint 0
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            print("Failed to compute camera-to-robot transform.")
            return

        # Use checkpoint 6 pure-vision pipeline to estimate cube pose
        result = get_transform_cube([cv_image, point_cloud], camera_intrinsic, t_cam_robot)
        if result is None:
            print("Cube not found.")
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
    
            # Pick up the cube using the pure-vision pose,
            # then drop it into the cup/basket using checkpoint 2 logic
            grasp_cube(arm, t_robot_cube)
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
