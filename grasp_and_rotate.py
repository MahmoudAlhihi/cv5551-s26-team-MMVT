import cv2
import numpy
from xarm.wrapper import XArmAPI
from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot

GRIPPER_LENGTH = 0.067 * 1000

CUBE_SIZE = 0.025 # 25 mm
ROBOT_IP = '192.168.1.182'

SAFE_Z_MM = 150.0 # safe travel height (mm)
GRASP_Z_OFFSET = 0.005 # 5 mm above cube bottom — gripper grips around mid-height
TRAVEL_SPEED = 200 # mm/s — free-space moves
GRASP_SPEED = 50 # mm/s — slow on descent to cube
LIFT_HEIGHT_MM = 30.0 # how high to lift cube before rotating (mm above contact)

GRIPPER_OPEN  = 850 # encoder counts — fully open
GRIPPER_CLOSE = 200 # encoder counts — firm grasp on 25 mm cube

ROTATE_ANGLE_DEG = 90.0 # default in plce rotation (deg)

# Cube Detection
def get_all_cube_poses(observation, camera_intrinsic, camera_pose):
    image, point_cloud = observation

    bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) if image.shape[2] == 4 else image
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    red_lo = cv2.inRange(hsv, numpy.array([0,   80,  80]), numpy.array([10,  255, 255]))
    red_hi = cv2.inRange(hsv, numpy.array([160, 80,  80]), numpy.array([180, 255, 255]))
    green = cv2.inRange(hsv, numpy.array([35,  50,  50]), numpy.array([80,  255, 255]))
    blue = cv2.inRange(hsv, numpy.array([90,  80,  60]), numpy.array([130, 255, 255]))

    full_mask = cv2.bitwise_or(cv2.bitwise_or(red_lo, red_hi), cv2.bitwise_or(green, blue))

    kernel = numpy.ones((5, 5), numpy.uint8)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN,  kernel)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(full_mask)

    T_robot_cam = camera_pose
    T_cam_robot = numpy.linalg.inv(T_robot_cam)
    detected = []

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 500:
            continue

        cube_mask = cv2.erode(
            (labels == i).astype(numpy.uint8),
            numpy.ones((3, 3), numpy.uint8), iterations=1
        )

        raw  = point_cloud[cube_mask > 0][:, :3]
        cpts = raw[numpy.all(numpy.isfinite(raw), axis=1)]
        if len(cpts) < 50:
            continue

        cpts_m = cpts / 1000.0
        ones = numpy.ones((len(cpts_m), 1))
        cpts_robot = (T_cam_robot @ numpy.hstack([cpts_m, ones]).T).T[:, :3]

        z_max   = numpy.max(cpts_robot[:, 2])
        top_pts = cpts_robot[cpts_robot[:, 2] > (z_max - 0.005)]
        if len(top_pts) < 10:
            continue

        rect = cv2.minAreaRect(top_pts[:, :2].astype(numpy.float32))
        (center_xy, (w, h), angle_deg) = rect
        if w < h:
            angle_deg += 90.0

        yaw_robot = numpy.deg2rad(angle_deg)
        center_z= z_max - (CUBE_SIZE / 2.0)

        Rz = numpy.array([
            [ numpy.cos(yaw_robot), -numpy.sin(yaw_robot), 0],
            [ numpy.sin(yaw_robot),  numpy.cos(yaw_robot), 0],
            [ 0,                     0,                    1]
        ]) @ numpy.diag([1, -1, -1])

        t_robot_cube = numpy.eye(4)
        t_robot_cube[:3, :3] = Rz
        t_robot_cube[:3,  3] = [center_xy[0], center_xy[1], center_z]

        t_cam_cube = T_robot_cam @ t_robot_cube
        detected.append((t_robot_cube, t_cam_cube))

    return detected

def cube_yaw_deg(t_robot_cube: numpy.ndarray) -> float:
    """Extract the yaw angle (degrees) of the cube in the robot XY plane"""
    Rz = t_robot_cube[:3, :3]
    yaw_rad = numpy.arctan2(Rz[1, 0], Rz[0, 0])
    return numpy.rad2deg(yaw_rad)


def set_gripper(arm: XArmAPI, position: int, wait: bool = True) -> None:
    """Open or close the gripper"""
    arm.set_gripper_position(position, wait=wait)

def grasp_and_rotate(arm: XArmAPI,
                     t_robot_cube: numpy.ndarray,
                     rotate_deg: float = ROTATE_ANGLE_DEG) -> None:
    """
    Grasp a cube and rotate it in place by `rotate_deg` degrees

    Steps
    1 Move above cube at safe height (gripper open)
    2 Descend to grasp height
    3 Close gripper
    4 Lift cube slightly off the table
    5 Rotate end-effector yaw by `rotate_deg`
    6 Lower cube back to table height
    7 Open gripper and retreat to safe height

    Params
    arm           : XArmAPI instance (already connected & enabled)
    t_robot_cube  : 4×4 pose of the cube in the robot frame (metres)
    rotate_deg    : Rotation to apply (degrees). Positive = CCW from above
    """
    cube_pos  = t_robot_cube[:3, 3] # metres
    yaw_now   = cube_yaw_deg(t_robot_cube)
    yaw_new   = yaw_now + rotate_deg

    # Grasp height: mid-cube, raised slightly so fingers clear the table
    grasp_z_mm = (cube_pos[2] - CUBE_SIZE / 2.0 + GRASP_Z_OFFSET) * 1000.0
    lift_z_mm  = grasp_z_mm + LIFT_HEIGHT_MM

    cx_mm = cube_pos[0] * 1000.0
    cy_mm = cube_pos[1] * 1000.0

    print(f"  Cube centre: ({cx_mm:.1f}, {cy_mm:.1f}, {cube_pos[2]*1000:.1f}) mm")
    print(f"  Cube yaw now: {yaw_now:.1f}°  →  target yaw: {yaw_new:.1f}°")
    print(f"  Grasp Z: {grasp_z_mm:.1f} mm")
    print(f"  Lift Z: {lift_z_mm:.1f} mm")

    # 1 — Move above cube (safe height, gripper open)
    print("[1/7] Moving above cube...")
    set_gripper(arm, GRIPPER_OPEN)
    arm.set_position(
        x=cx_mm, y=cy_mm, z=SAFE_Z_MM,
        roll=180, pitch=0, yaw=yaw_now,
        wait=True, speed=TRAVEL_SPEED, mvacc=500
    )

    # 2 — Descend to grasp height (keep same yaw)
    print("[2/7] Descending to grasp height...")
    arm.set_position(
        x=cx_mm, y=cy_mm, z=grasp_z_mm,
        roll=180, pitch=0, yaw=yaw_now,
        wait=True, speed=GRASP_SPEED, mvacc=200
    )

    # 3 — Close gripper
    print("[3/7] Closing gripper...")
    set_gripper(arm, GRIPPER_CLOSE)

    # 4 — Lift cube off table
    print("[4/7] Lifting cube...")
    arm.set_position(
        x=cx_mm, y=cy_mm, z=lift_z_mm,
        roll=180, pitch=0, yaw=yaw_now,
        wait=True, speed=GRASP_SPEED, mvacc=200
    )   

    # 5 — Rotate in place
    print(f"[5/7] Rotating {rotate_deg:+.1f}°...")
    arm.set_position(
        x=cx_mm, y=cy_mm, z=lift_z_mm,
        roll=180, pitch=0, yaw=yaw_new,
        wait=True, speed=GRASP_SPEED, mvacc=200
    )

    # 6 — Lower cube back to table
    print("[6/7] Lowering cube to table...")
    arm.set_position(
        x=cx_mm, y=cy_mm, z=grasp_z_mm,
        roll=180, pitch=0, yaw=yaw_new,
        wait=True, speed=GRASP_SPEED, mvacc=200
    )

    # 7 — Release and retreat
    print("[7/7] Opening gripper and retreating...")
    set_gripper(arm, GRIPPER_OPEN)
    arm.set_position(
        x=cx_mm, y=cy_mm, z=SAFE_Z_MM,
        roll=180, pitch=0, yaw=yaw_new,
        wait=True, speed=TRAVEL_SPEED, mvacc=500
    )

    print("Rotation complete")


#Entry point

def main():
    zed = ZedCamera()
    arm = XArmAPI(ROBOT_IP)
    arm.connect()
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)

    set_gripper(arm, GRIPPER_OPEN)

    try:
        cv_image, point_cloud = zed.get_synchronized_frame()
        t_cam_robot = get_transform_camera_robot(cv_image, zed.camera_intrinsic)
        if t_cam_robot is None:
            print("Could not compute camera-robot transform")
            return

        cube_results = get_all_cube_poses(
            [cv_image, point_cloud], zed.camera_intrinsic, t_cam_robot
        )

        if not cube_results:
            print("No cubes detected")
            return

        print(f"Detected {len(cube_results)} cube(s). Using the first one")

        # Visualise detections
        for _, t_cam in cube_results:
            draw_pose_axes(cv_image, zed.camera_intrinsic, t_cam)
        cv2.imshow('Detected Cubes', cv_image)
        if cv2.waitKey(0) != ord('k'):
            print("Aborted by user.")
            return

        rotate_deg = ROTATE_ANGLE_DEG # change to any angle in deg

        t_robot_cube, _ = cube_results[0]
        print(f"\nGrasping and rotating cube by {rotate_deg}°...")
        grasp_and_rotate(arm, t_robot_cube, rotate_deg=rotate_deg)

        arm.move_gohome(wait=True, speed=TRAVEL_SPEED, mvacc=500)

    finally:
        arm.disconnect()
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()