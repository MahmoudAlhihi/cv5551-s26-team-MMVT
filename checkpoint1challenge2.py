import cv2, numpy, time
from pupil_apriltags import Detector
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot

GRIPPER_LENGTH  = 0.067 * 1000
CUBE_TAG_FAMILY = 'tag36h11'
CUBE_TAG_ID     = 4
CUBE_TAG_SIZE   = 0.0206

robot_ip = '192.168.1.166'

# Reduced from 120 → 60 mm: less vertical travel per cycle while still
# clearing the tallest expected stack during transit
PRE_GRASP_HEIGHT = 100

# Fast transit between waypoints
TRANSIT_SPEED  = 600    # mm/s
TRANSIT_ACCEL  = 500   # mm/s²
# Slow, precise descent / ascent onto the cube
DESCEND_SPEED  = 600    # mm/s
DESCEND_ACCEL  = 500   # mm/s²


def grasp_cube_large(arm, cube_pose, size_m):
    x         = cube_pose[0, 3] * 1000
    y         = cube_pose[1, 3] * 1000
    z         = cube_pose[2, 3] * 1000   # center z in mm — same as working checkpoint1
    rot       = Rotation.from_matrix(cube_pose[:3, :3])
    r, p, yaw = rot.as_euler('xyz', degrees=False)

    arm.open_lite6_gripper()
    arm.set_position(x, y, -z + PRE_GRASP_HEIGHT, r, p, yaw,
                     is_radian=True, wait=True,
                     speed=TRANSIT_SPEED, mvacc=TRANSIT_ACCEL)
    arm.set_position(x, y, -z + (size_m * 1000 * 0.2 + 44), r, p, yaw,
                     is_radian=True, wait=True,
                     speed=DESCEND_SPEED, mvacc=DESCEND_ACCEL)
    arm.close_lite6_gripper()
    time.sleep(0.15)
    #arm.stop_lite6_gripper()
    arm.set_position(x, y, -z + PRE_GRASP_HEIGHT, r, p, yaw,
                     is_radian=True, wait=True,
                     speed=TRANSIT_SPEED, mvacc=TRANSIT_ACCEL)
    print(f"size={size_m*1000:.1f}mm | center_z={z:.1f}mm")


def grasp_cube_small(arm, cube_pose, size_m):
    x         = cube_pose[0, 3] * 1000
    y         = cube_pose[1, 3] * 1000
    z         = cube_pose[2, 3] * 1000   # center z in mm — same as working checkpoint1
    rot       = Rotation.from_matrix(cube_pose[:3, :3])
    r, p, yaw = rot.as_euler('xyz', degrees=False)

    arm.open_lite6_gripper()
    arm.set_position(x, y, -z + PRE_GRASP_HEIGHT, r, p, yaw,
                     is_radian=True, wait=True,
                     speed=TRANSIT_SPEED, mvacc=TRANSIT_ACCEL)
    arm.set_position(x, y, -z + (size_m * 1000 * 0.2 + 34), r, p, yaw,
                     is_radian=True, wait=True,
                     speed=DESCEND_SPEED, mvacc=DESCEND_ACCEL)
    arm.close_lite6_gripper()
    time.sleep(0.15)
    #arm.stop_lite6_gripper()
    arm.set_position(x, y, -z + PRE_GRASP_HEIGHT, r, p, yaw,
                     is_radian=True, wait=True,
                     speed=TRANSIT_SPEED, mvacc=TRANSIT_ACCEL)
    print(f"size={size_m*1000:.1f}mm | center_z={z:.1f}mm")

def place_cube(arm, cube_pose, size_m):
    """
    Release a cube at cube_pose (4×4, translations in metres).
    Uses the same descent offset as the grasp functions for consistent alignment.
    """
    x   = cube_pose[0, 3] * 1000
    y   = cube_pose[1, 3] * 1000
    z   = cube_pose[2, 3] * 1000
    rot = Rotation.from_matrix(cube_pose[:3, :3])
    r, p, yaw = rot.as_euler('xyz', degrees=False)

    place_z_offset = size_m * 1000 * 0.2 + 44   # mirrors grasp descent offset

    arm.set_position(x, y, -z + PRE_GRASP_HEIGHT, r, p, yaw,
                     is_radian=True, wait=True,
                     speed=TRANSIT_SPEED, mvacc=TRANSIT_ACCEL)
    arm.set_position(x, y, -z + place_z_offset, r, p, yaw,
                     is_radian=True, wait=True,
                     speed=DESCEND_SPEED, mvacc=DESCEND_ACCEL)
    arm.open_lite6_gripper()
    time.sleep(0.08)                                           # minimum for jaws to clear cube
    arm.stop_lite6_gripper()
    arm.set_position(x, y, -z + PRE_GRASP_HEIGHT, r, p, yaw,
                     is_radian=True, wait=True,
                     speed=TRANSIT_SPEED, mvacc=TRANSIT_ACCEL)


def get_transform_cube(observation, camera_intrinsic, camera_pose):
    """
    Detect the cube AprilTag and return (t_robot_cube, t_cam_cube).
    Returns None if the tag is not found.
    """
    if len(observation.shape) > 2 and observation.shape[2] == 4:
        gray = cv2.cvtColor(observation, cv2.COLOR_BGRA2GRAY)
    elif len(observation.shape) > 2:
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    else:
        gray = observation

    detector = Detector(families=CUBE_TAG_FAMILY)
    fx = camera_intrinsic[0, 0]
    fy = camera_intrinsic[1, 1]
    cx = camera_intrinsic[0, 2]
    cy = camera_intrinsic[1, 2]

    tags = detector.detect(gray, estimate_tag_pose=True,
                           camera_params=(fx, fy, cx, cy),
                           tag_size=CUBE_TAG_SIZE)

    cube_tag = next((t for t in tags if t.tag_id == CUBE_TAG_ID), None)
    if cube_tag is None:
        print(f'Cube tag (ID {CUBE_TAG_ID}) not detected.')
        return None

    t_cam_cube          = numpy.eye(4)
    t_cam_cube[:3, :3]  = cube_tag.pose_R
    t_cam_cube[:3,  3]  = cube_tag.pose_t.flatten()

    t_robot_cube = numpy.linalg.inv(camera_pose) @ t_cam_cube
    return t_robot_cube, t_cam_cube


def main():
    zed = ZedCamera()
    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)

    try:
        cv_image        = zed.image
        t_cam_robot     = get_transform_camera_robot(cv_image, zed.camera_intrinsic)
        if t_cam_robot is None:
            return

        result = get_transform_cube(cv_image, zed.camera_intrinsic, t_cam_robot)
        if result is None:
            print('Cube not found.')
            return
        t_robot_cube, t_cam_cube = result

        draw_pose_axes(cv_image, zed.camera_intrinsic, t_cam_cube)
        cv2.namedWindow('Verifying Cube Pose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Cube Pose', 1280, 720)
        cv2.imshow('Verifying Cube Pose', cv_image)

        if cv2.waitKey(0) == ord('k'):
            cv2.destroyAllWindows()
            grasp_cube_large(arm, t_robot_cube, CUBE_TAG_SIZE)
            place_cube(arm, t_robot_cube, CUBE_TAG_SIZE)

    finally:
        arm.move_gohome(wait=True)
        arm.disconnect()
        zed.close()


if __name__ == "__main__":
    main()