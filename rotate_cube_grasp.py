import cv2
import numpy
from xarm.wrapper import XArmAPI
from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1challenge1 import GRIPPER_LENGTH
import time

# Params
CUBE_SIZE = 0.025  # 25 mm
ROBOT_IP  = '192.168.1.182'

SAFE_Z_MM = 150.0   # safe travel height (mm)
GRASP_Z_FRAC = 0.50    # fraction of cube height to grasp at (0=bottom,1=top)
GRASP_SPEED = 80 # mm/s — approach speed
TRAVEL_SPEED = 200 # mm/s — free-space moves
ROTATE_SPEED = 30 # deg/s — wrist rotation speed
GRIPPER_OPEN = 850 # gripper open position (pulse)
GRIPPER_CLOSE = 280 # gripper closed position for 25 mm cube (pulse)
GRIPPER_SPEED = 5000 # gripper speed
LIFT_AFTER_MM = 5.0 # how much to lift before rotating (mm)


# Cube Detection

def get_all_cube_poses(observation, camera_intrinsic, camera_pose):
    image, point_cloud = observation

    bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) if image.shape[2] == 4 else image
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    red_lo = cv2.inRange(hsv, numpy.array([0,   80,  80]), numpy.array([10,  255, 255]))
    red_hi = cv2.inRange(hsv, numpy.array([160, 80,  80]), numpy.array([180, 255, 255]))
    green  = cv2.inRange(hsv, numpy.array([35,  50,  50]), numpy.array([80,  255, 255]))
    blue   = cv2.inRange(hsv, numpy.array([90,  80,  60]), numpy.array([130, 255, 255]))

    full_mask = cv2.bitwise_or(cv2.bitwise_or(red_lo, red_hi), cv2.bitwise_or(green, blue))

    kernel    = numpy.ones((5, 5), numpy.uint8)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN,  kernel)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(full_mask)

    T_robot_cam = camera_pose
    T_cam_robot = numpy.linalg.inv(T_robot_cam)
    detected    = []

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

        cpts_m     = cpts / 1000.0
        ones       = numpy.ones((len(cpts_m), 1))
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
        center_z  = z_max - (CUBE_SIZE / 2.0)

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


# Yaw helper

def _current_yaw(t_robot_cube):
    """Extract yaw (Z-rotation) from a 4×4 pose matrix (radians)."""
    return numpy.arctan2(t_robot_cube[1, 0], t_robot_cube[0, 0])


# Grasp-based rotation

def rotate_cube_by_grasping(arm, t_robot_cube, delta_yaw_deg):
    """
    Rotate a cube by delta_yaw_deg by:
      1. Opening the gripper
      2. Aligning the TCP yaw to the cube's current yaw
      3. Descending to grasp height above the cube centre
      4. Closing the gripper
      5. Lifting slightly
      6. Rotating the TCP yaw by delta_yaw_deg (in-place wrist rotation)
      7. Lowering back to the table
      8. Opening the gripper
      9. Retreating upward

    Parameters
    ----------
    arm           : XArmAPI instance
    t_robot_cube  : 4×4 cube pose in robot frame (metres)
    delta_yaw_deg : rotation to apply in degrees (+CCW, −CW viewed from above)
    """
    cube_x_mm = t_robot_cube[0, 3] * 1000.0
    cube_y_mm = t_robot_cube[1, 3] * 1000.0
    cube_z_mm = t_robot_cube[2, 3] * 1000.0  # cube centre Z

    cube_yaw_deg = numpy.rad2deg(_current_yaw(t_robot_cube))
    grasp_yaw_deg = cube_yaw_deg  # align gripper fingers with cube faces

    # Grasp height: midpoint of cube (GRASP_Z_FRAC from bottom)
    cube_bottom_mm = cube_z_mm - (CUBE_SIZE / 2.0) * 1000.0
    grasp_z_mm     = cube_bottom_mm + GRASP_Z_FRAC * CUBE_SIZE * 1000.0

    target_yaw_deg = grasp_yaw_deg + delta_yaw_deg

    print(f"\n── Grasp-rotate cube {delta_yaw_deg:+.1f}° ──")
    print(f"  Cube XY       : ({cube_x_mm:.1f}, {cube_y_mm:.1f}) mm")
    print(f"  Cube Z centre : {cube_z_mm:.1f} mm  |  Grasp Z: {grasp_z_mm:.1f} mm")
    print(f"  Gripper yaw   : {grasp_yaw_deg:.1f}° → {target_yaw_deg:.1f}°")

    # 1 Open gripper
    print("  [1/9] Opening gripper...")
    arm.set_gripper_position(GRIPPER_OPEN, speed=GRIPPER_SPEED, wait=True)

    # 2 Move above cube with gripper aligned to cube face
    print("  [2/9] Moving above cube (safe height)...")
    arm.set_position(
        x=cube_x_mm, y=cube_y_mm, z=SAFE_Z_MM,
        roll=180, pitch=0, yaw=grasp_yaw_deg,
        wait=True, speed=TRAVEL_SPEED, mvacc=500
    )

    # 3 Descend to grasp height
    print(f"  [3/9] Descending to grasp height ({grasp_z_mm:.1f} mm)...")
    arm.set_position(
        x=cube_x_mm, y=cube_y_mm, z=grasp_z_mm,
        roll=180, pitch=0, yaw=grasp_yaw_deg,
        wait=True, speed=GRASP_SPEED, mvacc=200
    )

    # 4 Close gripper on cube
    print("  [4/9] Closing gripper...")
    arm.set_gripper_position(GRIPPER_CLOSE, speed=GRIPPER_SPEED, wait=True)

    # 5 Lift slightly before rotating (reduces table friction)
    lift_z_mm = grasp_z_mm + LIFT_AFTER_MM
    print(f"  [5/9] Lifting {LIFT_AFTER_MM:.1f} mm before rotation...")
    arm.set_position(
        x=cube_x_mm, y=cube_y_mm, z=lift_z_mm,
        roll=180, pitch=0, yaw=grasp_yaw_deg,
        wait=True, speed=GRASP_SPEED, mvacc=200
    )

    # 6 Rotate TCP yaw in place
    print(f"  [6/9] Rotating wrist to {target_yaw_deg:.1f}°...")
    arm.set_position(
        x=cube_x_mm, y=cube_y_mm, z=lift_z_mm,
        roll=180, pitch=0, yaw=target_yaw_deg,
        wait=True, speed=ROTATE_SPEED, mvacc=100
    )

    # 7 Lower back to grasp height (set cube down)
    print("  [7/9] Lowering cube back to table...")
    arm.set_position(
        x=cube_x_mm, y=cube_y_mm, z=grasp_z_mm,
        roll=180, pitch=0, yaw=target_yaw_deg,
        wait=True, speed=GRASP_SPEED, mvacc=200
    )

    # 8 Open gripper to release
    print("  [8/9] Opening gripper (releasing cube)...")
    arm.set_gripper_position(GRIPPER_OPEN, speed=GRIPPER_SPEED, wait=True)

    # 9 Retreat upward
    print("  [9/9] Retreating to safe height...")
    arm.set_position(
        x=cube_x_mm, y=cube_y_mm, z=SAFE_Z_MM,
        roll=180, pitch=0, yaw=target_yaw_deg,
        wait=True, speed=TRAVEL_SPEED, mvacc=500
    )

    print("  Grasp-rotate complete.")


def rotate_cube_to_angle(arm, t_robot_cube, target_yaw_deg):
    """
    Rotate cube to a specific absolute yaw angle (single grasp attempt).

    Parameters
    ----------
    arm            : XArmAPI instance
    t_robot_cube   : 4×4 cube pose in robot frame (metres)
    target_yaw_deg : desired final yaw angle (degrees, robot frame)
    """
    current_yaw_deg = numpy.rad2deg(_current_yaw(t_robot_cube))
    delta = target_yaw_deg - current_yaw_deg
    delta = (delta + 180.0) % 360.0 - 180.0

    print(f"Current yaw: {current_yaw_deg:.1f}°  Target: {target_yaw_deg:.1f}°  Delta: {delta:.1f}°")
    rotate_cube_by_grasping(arm, t_robot_cube, delta)


def rotate_cube_to_angle_closed_loop(arm, zed, t_cam_robot, t_robot_cube_initial,
                                     target_yaw_deg,
                                     tolerance_deg=5.0, max_attempts=4):
    """
    Closed-loop grasp-rotate: grasp → rotate → re-detect → repeat until
    within tolerance or max_attempts is reached.

    Parameters
    ----------
    arm                  : XArmAPI instance
    zed                  : ZedCamera instance
    t_cam_robot          : 4×4 camera-to-robot transform
    t_robot_cube_initial : initial 4×4 cube pose (metres)
    target_yaw_deg       : desired final yaw (degrees, robot frame)
    tolerance_deg        : stop when |error| < this (degrees)
    max_attempts         : maximum grasp-rotate iterations
    """
    t_robot_cube = t_robot_cube_initial.copy()

    for attempt in range(1, max_attempts + 1):
        current_yaw_deg = numpy.rad2deg(_current_yaw(t_robot_cube))
        delta = target_yaw_deg - current_yaw_deg
        delta = (delta + 180.0) % 360.0 - 180.0

        print(f"\n── Attempt {attempt}/{max_attempts} ──")
        print(f"  Current yaw: {current_yaw_deg:.1f}°  Target: {target_yaw_deg:.1f}°  Error: {delta:.1f}°")

        if abs(delta) < tolerance_deg:
            print(f"  Within tolerance ({tolerance_deg}°). Done.")
            return t_robot_cube

        rotate_cube_by_grasping(arm, t_robot_cube, delta)

        # Re detect
        print("  Re-detecting cube...")
        arm.move_gohome(wait=True)
        cv_image, point_cloud = zed.get_synchronized_frame()
        results = get_all_cube_poses(
            [cv_image, point_cloud], zed.camera_intrinsic, t_cam_robot
        )

        if not results:
            print("  WARNING: Could not re-detect cube. Stopping.")
            return t_robot_cube

        prev_xy = t_robot_cube[:2, 3]
        best    = min(results, key=lambda r: numpy.linalg.norm(r[0][:2, 3] - prev_xy))
        t_robot_cube, _ = best

    print(f"\nMax attempts ({max_attempts}) reached.")
    return t_robot_cube


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    zed = ZedCamera()
    arm = XArmAPI(ROBOT_IP)
    arm.connect()
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)

    # Enable gripper
    arm.set_gripper_enable(True)
    time.sleep(0.5)
    arm.set_gripper_mode(0)
    time.sleep(0.5)
    arm.set_gripper_speed(GRIPPER_SPEED)
    time.sleep(0.5)

    arm.move_gohome(wait=True)


    try:
        cv_image, point_cloud = zed.get_synchronized_frame()
        t_cam_robot = get_transform_camera_robot(cv_image, zed.camera_intrinsic)
        if t_cam_robot is None:
            print("Could not compute camera-robot transform.")
            return

        cube_results = get_all_cube_poses(
            [cv_image, point_cloud], zed.camera_intrinsic, t_cam_robot
        )

        if not cube_results:
            print("No cubes detected.")
            return

        print(f"Detected {len(cube_results)} cube(s).")

        # Visualise detections
        for _, t_cam in cube_results:
            draw_pose_axes(cv_image, zed.camera_intrinsic, t_cam)
        cv2.imshow('Detected Cubes', cv_image)
        key = cv2.waitKey(0)
        if key != ord('k'):
            print("Aborted by user.")
            return

        t_robot_cube, _ = cube_results[0]
        current_yaw = numpy.rad2deg(_current_yaw(t_robot_cube))
        print(f"\nCube detected. Current yaw: {current_yaw:.1f}°")

        # ── Choose one of the options below ──────────────────────────────

        # Option A: rotate by a fixed delta (single grasp, open loop)
        print("\n[Option A] Grasping and rotating +45°...")
        rotate_cube_by_grasping(arm, t_robot_cube, delta_yaw_deg=45.0)

        # Option B: rotate to an absolute angle (single grasp, open loop)
        # print("\n[Option B] Grasping and rotating to 0°...")
        # rotate_cube_to_angle(arm, t_robot_cube, target_yaw_deg=0.0)

        # Option C: closed-loop grasp-rotate to an absolute angle
        # print("\n[Option C] Closed-loop grasp-rotate to 90°...")
        # rotate_cube_to_angle_closed_loop(
        #     arm, zed, t_cam_robot, t_robot_cube,
        #     target_yaw_deg=90.0,
        #     tolerance_deg=5.0,
        #     max_attempts=4
        # )

        arm.move_gohome(wait=True)

    finally:
        arm.set_gripper_position(GRIPPER_OPEN, speed=GRIPPER_SPEED, wait=True)
        arm.disconnect()
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()