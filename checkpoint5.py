from checkpoint3 import CubePoseDetector

import cv2, time
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH
from checkpoint4 import STACK_HEIGHT

stacking_order = ['red cube', 'green cube', 'blue cube']   # From top to bottom
robot_ip = '192.168.1.166'

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

        # TODO
        # compute camera to robot transform
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            print("Wsp registration failed")
            return

        cube_pose_detector.camera_pose = t_cam_robot

        # detect all cubes in stacking order
        poses = {}
        for cube in stacking_order:
            result = cube_pose_detector.get_transforms(cv_image, cube)
            if result is None:
                print(f"Could not detect {cube}")
                return
            t_robot, t_cam = result
            poses[cube] = (t_robot, t_cam)

        # visualize all detected cubes
        for cube in stacking_order:
            _, t_cam = poses[cube]
            draw_pose_axes(cv_image, camera_intrinsic, t_cam)

        cv2.imshow('Checkpoint 5', cv_image)
        print("Press 'k' to execute stacking")
        if cv2.waitKey(0) != ord('k'):
            return
        cv2.destroyAllWindows()

        # stacking logic (bottom → top)
        base_cube = stacking_order[-1]
        t_robot_base, _ = poses[base_cube]

        # stack each cube on top of base
        for i in reversed(range(len(stacking_order) - 1)):
            cube = stacking_order[i]
            t_robot_cube, _ = poses[cube]

            # pick cube
            grasp_cube(arm, t_robot_cube)
            time.sleep(0.5)

            # compute stacking height
            t_robot_stack = t_robot_base.copy()
            t_robot_stack[2, 3] -= STACK_HEIGHT #Check in labs

            # align orientation
            t_robot_stack[:3, :3] = t_robot_base[:3, :3]

            # place cube
            place_cube(arm, t_robot_stack)

            # update base for next stacking level
            t_robot_base = t_robot_stack.copy()
            time.sleep(0.5)
    
    finally:
        # Close Lite6 Robot
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()