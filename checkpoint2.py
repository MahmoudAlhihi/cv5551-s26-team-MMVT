import cv2, numpy, time
from xarm.wrapper import XArmAPI
import math
from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, get_transform_cube, GRIPPER_LENGTH

# TODO
BASKET_POSE = [228.1, -303.6, 23.2, 178.2, -3.8, 4.1 ] # Measure it using the robot's free drive mode.

BASKET_APPROACH_HEIGHT = 0.05 # height(metres) to approach above the basket before releasing

robot_ip = '192.168.1.158'

def place_in_basket(arm, basket_pose, vaccum_gripper=False):
    """
    Move the robot arm to the basket location and release the grasped object.

    Parameters
    ----------
    arm : xarm.wrapper.XArmAPI
        The initialized XArm API object contring the Lite6 robot.
    basket_pose : list or numpy.ndarray
        A 6-element array representing the target drop-off pose in the robot 
        base frame formatted as [x, y, z, r, p, y]. 
        Translational units (x, y, z) are in meters, and rotational units 
        (r, p, y) are in radians.
    vaccum_gripper : bool, optional
        If True, uses the vacuum gripper logic instead of the standard Lite6 
        gripper. Defaults to False.
    """
    # TODO
    # convert metres to mm for xArm API
    x = basket_pose[0] * 1000
    y = basket_pose[1] * 1000
    z = basket_pose[2] * 1000
    r = basket_pose[3] * (2*math.pi)/180
    p = basket_pose[4] * (2*math.pi)/180
    yaw = basket_pose[5] * (2*math.pi)/180
 
    # approach from a safe height above the basket so we clear the cup lip
    z_offset = z + BASKET_APPROACH_HEIGHT * 1000
    arm.set_position(x, y, z_offset, r, p, yaw, is_radian=True, wait=True)
    time.sleep(0.3)
 
    # release objecty
    if vaccum_gripper:
        arm.set_vacuum_gripper(False, wait=True)
    else:
        arm.open_lite6_gripper()
        time.sleep(0.5)
        arm.stop_lite6_gripper()
 
    # Retreat back up so we don't knock the cupg
    arm.set_position(x, y, z_offset + 30, r, p, yaw, is_radian=True, wait=True)
    time.sleep(0.3)

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

        t_cam_cube = None
        #TODO
        # capture image and compute camera to robot transform
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            print('Failed to compute cam-robot transform')
            return

        # detect cube pose
        result = get_transform_cube(cv_image, camera_intrinsic, t_cam_robot)
        if result is None:
            print('Cube not found')
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
            # pick up the cube, then drop it in the basket
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

