import cv2
import numpy
from xarm.wrapper import XArmAPI
from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1challenge1 import GRIPPER_LENGTH

# Params
CUBE_SIZE = 0.025 # 25 mm
ROBOT_IP  = '192.168.1.182'

SAFE_Z_MM = 150.0  # safe travel height (mm)
APPROACH_OFFSET = 0.07   # 7 cm behind contact point
PUSH_DISTANCE = 0.06   # 6 cm forward push (translating)
PUSH_SPEED = 60  # mm/s — slow for controlled contact
TRAVEL_SPEED = 200 # mm/s — free-space moves

ROTATE_PUSH_SPEED = 40  # mm/s — slower for rotation accuracy
ROTATE_TRAVEL_SPEED = 200 # mm/s
CUBE_HALF = CUBE_SIZE / 2.0 # 12.5 mm
ROTATE_APPROACH_OFFSET = 0.06 # 6 cm clearance before contact face


# ── Cube Detection

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


# Translation / push helpers

def compute_push_direction(t_robot_cube, target_xy_mm):
    """
    Return a unit push direction vector pointing from the cube toward target_xy_mm.  Defaults to robot +X if target is None.
    """
    if target_xy_mm is None:
        return numpy.array([1.0, 0.0])

    cube_xy = t_robot_cube[:2, 3] * 1000.0
    target  = numpy.array(target_xy_mm, dtype=float)
    vec     = target - cube_xy
    norm    = numpy.linalg.norm(vec)
    if norm < 1e-6:
        return numpy.array([1.0, 0.0])
    return vec / norm


def push_cube(arm, t_robot_cube, target_xy_mm=None,
              push_distance_m=PUSH_DISTANCE):
    """
    Translate a cube toward target_xy_mm (or along +X if None).

    Parameters
    ----------
    arm: XArmAPI instance
    t_robot_cube: 4×4 cube pose in robot frame (metres)
    target_xy_mm: [x, y] target in mm; None → push along robot +X
    push_distance_m: distance (m) the gripper travels through the cube
    """
    cube_pos = t_robot_cube[:3, 3]
    dx, dy   = compute_push_direction(t_robot_cube, target_xy_mm)

    if target_xy_mm is not None:
        cube_xy_mm = cube_pos[:2] * 1000.0
        target     = numpy.array(target_xy_mm, dtype=float)
        dist_mm    = numpy.linalg.norm(target - cube_xy_mm)
        push_distance_m = (dist_mm / 1000.0) + 0.005   # 5 mm overshoot

    contact_z_mm = (cube_pos[2] - CUBE_SIZE * 0.05) * 1000.0

    approach_x = (cube_pos[0] - dx * APPROACH_OFFSET) * 1000.0
    approach_y = (cube_pos[1] - dy * APPROACH_OFFSET) * 1000.0
    end_x = (cube_pos[0] + dx * push_distance_m) * 1000.0
    end_y = (cube_pos[1] + dy * push_distance_m) * 1000.0

    print(f"Cube centre: ({cube_pos[0]*1000:.1f}, {cube_pos[1]*1000:.1f}) mm")
    print(f"Push direction: ({dx:.3f}, {dy:.3f})")
    print(f"Approach: ({approach_x:.1f}, {approach_y:.1f}) mm")
    print(f"End: ({end_x:.1f}, {end_y:.1f}) mm")
    print(f"Contact Z: {contact_z_mm:.1f} mm")

    print("  [1/4] Moving above approach point...")
    arm.set_position(
        x=approach_x, y=approach_y, z=SAFE_Z_MM,
        roll=180, pitch=0, yaw=90,
        wait=True, speed=TRAVEL_SPEED, mvacc=500
    )

    print("  [2/4] Descending to contact height...")
    arm.set_position(
        x=approach_x, y=approach_y, z=contact_z_mm,
        roll=180, pitch=0, yaw=90,
        wait=True, speed=100, mvacc=200
    )

    print("  [3/4] Pushing cube...")
    arm.set_position(
        x=end_x, y=end_y, z=contact_z_mm,
        roll=180, pitch=0, yaw=90,
        wait=True, speed=PUSH_SPEED, mvacc=200
    )

    print("  [4/4] Lifting clear...")
    arm.set_position(
        x=end_x, y=end_y, z=SAFE_Z_MM,
        roll=180, pitch=0, yaw=90,
        wait=True, speed=TRAVEL_SPEED, mvacc=500
    )

    print("  Push complete.")


# Rot helpers

def _rotation_matrix_2d(angle_rad):
    """2-D rotation matrix for angle_rad."""
    c, s = numpy.cos(angle_rad), numpy.sin(angle_rad)
    return numpy.array([[c, -s], [s, c]])


def _current_yaw(t_robot_cube):
    """Extract yaw (Z-rotation) from a 4×4 pose matrix (radians)."""
    return numpy.arctan2(t_robot_cube[1, 0], t_robot_cube[0, 0])


def _push_params_for_rotation(t_robot_cube, delta_yaw, push_depth_m=0.04):
    """
    Compute the contact point and push direction for an off-centre push
    that rotates the cube by delta_yaw radians in place.

    The contact is placed at ±half_size along the cube's local-Y axis on
    the face that opposes the push direction, so the gripper applies a
    moment about the cube's centre.

    Returns
    -------
    contact_xy_m: numpy array (2,) — contact point in robot frame, metres
    push_dir: numpy array (2,) — unit push direction in robot frame
    """
    centre   = t_robot_cube[:2, 3]
    cube_yaw = _current_yaw(t_robot_cube)

    # Push along cube local-X; flip direction if rotating CW
    push_dir_local = numpy.array([1.0, 0.0])
    push_dir = _rotation_matrix_2d(cube_yaw) @ push_dir_local
    if delta_yaw < 0:
        push_dir = -push_dir

    # Contact offset: ±half_size along cube local-Y
    # Positive delta_yaw (CCW) → contact above cube centre (local +Y)
    sign = +1.0 if delta_yaw >= 0 else -1.0
    local_y_dir = _rotation_matrix_2d(cube_yaw) @ numpy.array([0.0, sign])
    contact_offset = CUBE_HALF * local_y_dir

    # Contact sits on the face opposite to the push direction
    face_offset = -push_dir * CUBE_HALF
    contact_xy = centre + face_offset + contact_offset

    return contact_xy, push_dir


def rotate_cube_in_place(arm, t_robot_cube, delta_yaw_deg,
                         push_depth_m=0.04,
                         contact_z_frac=1.5):
    """
    Rotate a cube approximately delta_yaw_deg degrees in place using
    a single off-centre push.

    Parameters
    ----------
    arm: XArmAPI instance
    t_robot_cube: 4×4 cube pose in robot frame (metres)
    delta_yaw_deg: desired rotation in degrees (+CCW, −CW viewed from above)
    push_depth_m: how far the gripper travels past the contact point (m)
    contact_z_frac: height of contact as a fraction of CUBE_SIZE from the
                      bottom face (0 = base, 0.5 = mid, 1 = top).
                      Default 0.35 keeps the gripper below the top face.
    """
    delta_yaw  = numpy.deg2rad(delta_yaw_deg)
    contact_xy, push_dir = _push_params_for_rotation(
        t_robot_cube, delta_yaw, push_depth_m
    )

    cx_mm = contact_xy[0] * 1000.0
    cy_mm = contact_xy[1] * 1000.0
    dx, dy = push_dir

    # Z: cube bottom + fractional height
    cube_bottom_m  = t_robot_cube[2, 3] - CUBE_HALF
    contact_z_mm   = (cube_bottom_m + contact_z_frac * CUBE_SIZE) * 1000.0

    approach_x = cx_mm - dx * ROTATE_APPROACH_OFFSET * 1000.0
    approach_y = cy_mm - dy * ROTATE_APPROACH_OFFSET * 1000.0
    end_x      = cx_mm + dx * push_depth_m * 1000.0
    end_y      = cy_mm + dy * push_depth_m * 1000.0

    print(f"Rotating cube {delta_yaw_deg:+.1f}°")
    print(f"  Contact point : ({cx_mm:.1f}, {cy_mm:.1f}) mm")
    print(f"  Push direction: ({dx:.3f}, {dy:.3f})")
    print(f"  Contact Z     : {contact_z_mm:.1f} mm")
    print(f"  Approach      : ({approach_x:.1f}, {approach_y:.1f}) mm")
    print(f"  End           : ({end_x:.1f}, {end_y:.1f}) mm")

    print("  [1/4] Moving above approach point...")
    arm.set_position(
        x=approach_x, y=approach_y, z=SAFE_Z_MM,
        roll=180, pitch=0, yaw=90,
        wait=True, speed=ROTATE_TRAVEL_SPEED, mvacc=500
    )

    print("  [2/4] Descending to contact height...")
    arm.set_position(
        x=approach_x, y=approach_y, z=contact_z_mm,
        roll=180, pitch=0, yaw=90,
        wait=True, speed=100, mvacc=200
    )

    print("  [3/4] Pushing (rotating)...")
    arm.set_position(
        x=end_x, y=end_y, z=contact_z_mm,
        roll=180, pitch=0, yaw=90,
        wait=True, speed=ROTATE_PUSH_SPEED, mvacc=200
    )

    print("  [4/4] Lifting clear...")
    arm.set_position(
        x=end_x, y=end_y, z=SAFE_Z_MM,
        roll=180, pitch=0, yaw=90,
        wait=True, speed=ROTATE_TRAVEL_SPEED, mvacc=500
    )

    print("  Rotation push complete.")


def rotate_cube_to_angle(arm, t_robot_cube, target_yaw_deg, push_depth_m=0.04):
    """
    Rotate a cube to a specific absolute yaw angle in the robot frame.

    Makes one push; re-detect and call again to close the loop if needed.

    Parameters
    ----------
    arm: XArmAPI instance
    t_robot_cube: 4×4 cube pose in robot frame (metres)
    target_yaw_deg: desired final yaw angle (degrees, robot frame)
    push_depth_m: how far the gripper travels past the contact point (m)
    """
    current_yaw_deg = numpy.rad2deg(_current_yaw(t_robot_cube))
    delta           = target_yaw_deg - current_yaw_deg

    # Normalise delta to (−180, +180]
    delta = (delta + 180.0) % 360.0 - 180.0

    print(f"Current yaw: {current_yaw_deg:.1f}°")
    print(f"Target yaw: {target_yaw_deg:.1f}°")
    print(f"Delta: {delta:.1f}°")

    rotate_cube_in_place(arm, t_robot_cube, delta, push_depth_m=push_depth_m)


def rotate_cube_to_angle_closed_loop(arm, zed, t_cam_robot, t_robot_cube_initial, target_yaw_deg, push_depth_m=0.04, tolerance_deg=5.0, max_attempts=4):
    """
    Closed-loop version: push -> re-detect -> check error -> repeat until
    within tolerance or max_attempts is reached.

    Parameters
    ----------
    arm: XArmAPI instance
    zed: ZedCamera instance
    t_cam_robot: 4×4 camera-to-robot transform
    t_robot_cube_initial: initial 4×4 cube pose (metres)
    target_yaw_deg: desired final yaw angle (degrees, robot frame)
    push_depth_m: push depth per attempt (m)
    tolerance_deg: stop when |error| < this value (degrees)
    max_attempts: maximum number of push-and-check iterations
    """
    t_robot_cube = t_robot_cube_initial.copy()

    for attempt in range(1, max_attempts + 1):
        current_yaw_deg = numpy.rad2deg(_current_yaw(t_robot_cube))
        delta = target_yaw_deg - current_yaw_deg
        delta = (delta + 180.0) % 360.0 - 180.0

        print(f"\n── Attempt {attempt}/{max_attempts} ──")
        print(f"  Current yaw : {current_yaw_deg:.1f}°  Target: {target_yaw_deg:.1f}°  Error: {delta:.1f}°")

        if abs(delta) < tolerance_deg:
            print(f"  Within tolerance ({tolerance_deg}°). Done.")
            return t_robot_cube

        rotate_cube_in_place(arm, t_robot_cube, delta, push_depth_m=push_depth_m)

        # Re-detect
        print("  Re-detecting cube...")
        arm.move_gohome(wait=True)
        cv_image, point_cloud = zed.get_synchronized_frame()
        results = get_all_cube_poses(
            [cv_image, point_cloud], zed.camera_intrinsic, t_cam_robot
        )

        if not results:
            print("  WARNING: Could not re-detect cube after push. Stopping.")
            return t_robot_cube

        # Use the closest cube to the previous position
        prev_xy = t_robot_cube[:2, 3]
        best    = min(results,
                      key=lambda r: numpy.linalg.norm(r[0][:2, 3] - prev_xy))
        t_robot_cube, _ = best

    print(f"\nMax attempts ({max_attempts}) reached.")
    return t_robot_cube

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

        # ── Choose one of the options below

        # Option A: rotate by a fixed delta (single push, open loop)
        print("\n[Option A] Rotating +45° in place...")
        rotate_cube_in_place(arm, t_robot_cube, delta_yaw_deg=45.0)

        # Option B: rotate to an absolute angle (single push, open loop)
        # print("\n[Option B] Rotating to 0°...")
        # rotate_cube_to_angle(arm, t_robot_cube, target_yaw_deg=0.0)

        # Option C: closed-loop rotation to an absolute angle (push + re-detect)
        # print("\n[Option C] Closed-loop rotation to 90°...")
        # rotate_cube_to_angle_closed_loop(
        #     arm, zed, t_cam_robot, t_robot_cube,
        #     target_yaw_deg=90.0,
        #     tolerance_deg=5.0,
        #     max_attempts=4
        # )

        arm.move_gohome(wait=True)

    finally:
        arm.disconnect()
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()