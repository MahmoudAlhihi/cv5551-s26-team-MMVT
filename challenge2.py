import cv2
import numpy
import time
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot

# ── Constants ─────────────────────────────────────────────────────────────────
STACK_BASE      = [229.4, -296.4, 23.2]  # mm, fixed stacking XY footprint
PLACE_MARGIN_M  = 0.001                  # 1 mm inter-cube gap
CUBE_SIZE       = 0.025                  # 25 mm
ROBOT_IP        = '192.168.1.183'
GRIPPER_LENGTH  = 0.067 * 1000           # mm

PRE_GRASP_HEIGHT = 45    # mm — reduced from 60 to cut vertical travel
GRASP_HEIGHT     = 40    # mm descent offset for grasping
PLACE_HEIGHT     = 42    # mm descent offset for placing

TRANSIT_SPEED  = 900     # mm/s — fast travel between waypoints
TRANSIT_ACCEL  = 2500    # mm/s²
DESCEND_SPEED  = 350     # mm/s — slow, precise descent/ascent
DESCEND_ACCEL  = 1000    # mm/s²


# ── Grasp ─────────────────────────────────────────────────────────────────────
def grasp_cube(arm, cube_pose):
    """
    Pick a cube at cube_pose (4x4, translations in metres).

    open_lite6_gripper() is non-blocking — fires the command and returns
    immediately, so the gripper opens during transit to PRE_GRASP_HEIGHT.
    """
    x   = cube_pose[0, 3] * 1000
    y   = cube_pose[1, 3] * 1000
    z   = cube_pose[2, 3] * 1000
    rot = Rotation.from_matrix(cube_pose[:3, :3])
    r, p, yaw = rot.as_euler('xyz', degrees=False)

    arm.open_lite6_gripper()                               # async — opens in transit
    arm.set_position(x, y, -z + PRE_GRASP_HEIGHT, r, p, yaw,
                     is_radian=True, wait=True,
                     speed=TRANSIT_SPEED, mvacc=TRANSIT_ACCEL)
    arm.set_position(x, y, -z + GRASP_HEIGHT, r, p, yaw,
                     is_radian=True, wait=True,
                     speed=DESCEND_SPEED, mvacc=DESCEND_ACCEL)
    arm.close_lite6_gripper()
    time.sleep(0.15)                                       # minimum for jaws to seat
    arm.stop_lite6_gripper()                               # lock torque — prevents drift
    arm.set_position(x, y, -z + PRE_GRASP_HEIGHT, r, p, yaw,
                     is_radian=True, wait=True,
                     speed=TRANSIT_SPEED, mvacc=TRANSIT_ACCEL)


# ── Place ─────────────────────────────────────────────────────────────────────
def place_cube(arm, cube_pose):
    """
    Release a cube at cube_pose (4x4, translations in metres).
    """
    x   = cube_pose[0, 3] * 1000
    y   = cube_pose[1, 3] * 1000
    z   = cube_pose[2, 3] * 1000
    rot = Rotation.from_matrix(cube_pose[:3, :3])
    r, p, yaw = rot.as_euler('xyz', degrees=False)

    arm.set_position(x, y, -z + PRE_GRASP_HEIGHT, r, p, yaw,
                     is_radian=True, wait=True,
                     speed=TRANSIT_SPEED, mvacc=TRANSIT_ACCEL)
    arm.set_position(x, y, -z + PLACE_HEIGHT, r, p, yaw,
                     is_radian=True, wait=True,
                     speed=DESCEND_SPEED, mvacc=DESCEND_ACCEL)
    arm.open_lite6_gripper()
    time.sleep(0.08)                                       # minimum for jaws to clear cube
    arm.stop_lite6_gripper()
    arm.set_position(x, y, -z + PRE_GRASP_HEIGHT, r, p, yaw,
                     is_radian=True, wait=True,
                     speed=TRANSIT_SPEED, mvacc=TRANSIT_ACCEL)


# ── Cube Detection ────────────────────────────────────────────────────────────
def get_all_cube_poses(observation, camera_intrinsic, camera_pose):
    """
    Detect all coloured cubes and return a list of (t_robot_cube, t_cam_cube)
    using a top-surface centroid + minAreaRect yaw method.
    """
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

        rect                         = cv2.minAreaRect(top_pts[:, :2].astype(numpy.float32))
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

        t_robot_cube          = numpy.eye(4)
        t_robot_cube[:3, :3]  = Rz
        t_robot_cube[:3,  3]  = [center_xy[0], center_xy[1], center_z]

        t_cam_cube = T_robot_cam @ t_robot_cube
        detected.append((t_robot_cube, t_cam_cube))

    return detected


# ── Proximity Sort ────────────────────────────────────────────────────────────
def sort_by_proximity(cube_results):
    """
    Sort cubes by ascending XY distance to STACK_BASE so the arm picks
    the nearest cube first, minimising transit distance and collision risk.
    """
    base_xy = numpy.array([STACK_BASE[0] / 1000.0, STACK_BASE[1] / 1000.0])
    return sorted(
        cube_results,
        key=lambda r: numpy.linalg.norm(r[0][:2, 3] - base_xy)
    )


# ── Stacking ──────────────────────────────────────────────────────────────────
def stack_cubes(arm, cube_results):
    """
    Stack all detected cubes closest-first.

    Cube 0 (post-sort) becomes the base at STACK_BASE.
    Each subsequent cube is placed on the growing tower with a dynamically
    computed target Z.

    NOTE: There is no explicit post-place lift here.  grasp_cube() opens the
    gripper and moves to PRE_GRASP_HEIGHT as its very first step, which serves
    as the lift — removing a redundant vertical move per cube.
    """
    if not cube_results:
        print("No cubes detected.")
        return

    cube_results = sort_by_proximity(cube_results)
    n = len(cube_results)
    print(f"\nStacking {n} cube(s), closest → furthest from base:")

    # Fixed XY + rotation anchor for the whole stack
    base_t       = cube_results[0][0].copy()
    base_t[0, 3] = STACK_BASE[0] / 1000.0
    base_t[1, 3] = STACK_BASE[1] / 1000.0
    base_t[2, 3] = STACK_BASE[2] / 1000.0

    # ── Cube 0: place at base ──────────────────────────────────────────────────
    print(f"\n── Cube 1/{n} (BASE) ──")
    grasp_cube(arm, cube_results[0][0])
    place_cube(arm, base_t)

    z_top = base_t[2, 3] - CUBE_SIZE / 2.0   # top face of base cube (robot Z convention)

    # ── Remaining cubes ────────────────────────────────────────────────────────
    for i in range(1, n):
        t_robot_cube, _ = cube_results[i]
        print(f"\n── Cube {i+1}/{n} ──")

        grasp_cube(arm, t_robot_cube)

        # Target: margin above current top + this cube's half-height
        target_z             = z_top - PLACE_MARGIN_M - CUBE_SIZE / 2.0
        stack_target         = base_t.copy()
        stack_target[2, 3]   = target_z
        stack_target[:3, :3] = base_t[:3, :3]   # keep base orientation throughout

        place_cube(arm, stack_target)

        # No explicit post-place lift — next grasp_cube() starts with open_gripper
        # + move to PRE_GRASP_HEIGHT, which naturally clears the stack.

        z_top = target_z - CUBE_SIZE / 2.0      # advance top face for next cube

    print("\nAll cubes stacked.")


# ── Entry Point ───────────────────────────────────────────────────────────────
def main():
    zed = ZedCamera()
    arm = XArmAPI(ROBOT_IP)
    arm.connect()
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
        for _, t_cam in cube_results:
            draw_pose_axes(cv_image, zed.camera_intrinsic, t_cam)
        cv2.imshow('Detected Cubes', cv_image)
        if cv2.waitKey(0) != ord('k'):
            return

        stack_cubes(arm, cube_results)
        arm.move_gohome(wait=True)

    finally:
        arm.disconnect()
        zed.close()


if __name__ == "__main__":
    main()