import cv2
import numpy
import time
from xarm.wrapper import XArmAPI
from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoints.checkpoint0 import get_transform_camera_robot
from checkpoints.checkpoint1challenge1 import grasp_cube, place_cube, GRIPPER_LENGTH

STACK_BASE    = [229.4, -296.4, 23.2]   # mm, fixed stacking XY footprint
PLACE_MARGIN_M = 0.001                  # 1 mm gap between cubes
CUBE_SIZE      = 0.025                  # 25 mm
ROBOT_IP       = '192.168.1.183'

# Cube Detection
def get_all_cube_poses(observation, camera_intrinsic, camera_pose):
    """
    Detect all coloured cubes and return a list of (t_robot_cube, t_cam_cube)
    using a top-surface centroid + minAreaRect yaw method.
    """
    image, point_cloud = observation

    # Convert to BGR if BGRA
    bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) if image.shape[2] == 4 else image
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Colour masks
    red_lo  = cv2.inRange(hsv, numpy.array([0,   80,  80]), numpy.array([10,  255, 255]))
    red_hi  = cv2.inRange(hsv, numpy.array([160, 80,  80]), numpy.array([180, 255, 255]))
    green   = cv2.inRange(hsv, numpy.array([35,  50,  50]), numpy.array([80,  255, 255]))
    blue    = cv2.inRange(hsv, numpy.array([90,  80,  60]), numpy.array([130, 255, 255]))

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

        raw    = point_cloud[cube_mask > 0][:, :3]
        cpts   = raw[numpy.all(numpy.isfinite(raw), axis=1)]
        if len(cpts) < 50:
            continue

        # Transform to robot frame
        cpts_m      = cpts / 1000.0
        ones        = numpy.ones((len(cpts_m), 1))
        cpts_robot  = (T_cam_robot @ numpy.hstack([cpts_m, ones]).T).T[:, :3]

        # Top-surface centroid
        z_max      = numpy.max(cpts_robot[:, 2])
        top_pts    = cpts_robot[cpts_robot[:, 2] > (z_max - 0.005)]
        if len(top_pts) < 10:
            continue

        rect              = cv2.minAreaRect(top_pts[:, :2].astype(numpy.float32))
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


# Sorting 
def sort_by_proximity(cube_results):
    """
    Sort cubes closest to STACK_BASE first so the arm travels the
    shortest distance on the first pick and avoids knocking neighbours.
    """
    base_xy = numpy.array([STACK_BASE[0] / 1000.0, STACK_BASE[1] / 1000.0])
    return sorted(
        cube_results,
        key=lambda r: numpy.linalg.norm(r[0][:2, 3] - base_xy)
    )


# Stacking
def stack_cubes(arm, cube_results):
    """
    Stack all detected cubes, closest to STACK_BASE first.
    Cube 0 (after sorting) becomes the base; every subsequent cube
    is placed on the growing tower with a dynamically computed target Z.
    """
    if not cube_results:
        print("No cubes detected.")
        return

    cube_results = sort_by_proximity(cube_results)
    n = len(cube_results)
    print(f"\nStacking {n} cube(s), closest → furthest from base:")

    # Fixed XY/rotation anchor for the entire stack
    base_t          = cube_results[0][0].copy()
    base_t[0, 3]    = STACK_BASE[0] / 1000.0
    base_t[1, 3]    = STACK_BASE[1] / 1000.0
    base_t[2, 3]    = STACK_BASE[2] / 1000.0

    # ── Cube 0: move to base position ─────────────────────────────────────────
    print(f"\n── Cube 1/{n} (BASE) ──")
    grasp_cube(arm, cube_results[0][0])
    place_cube(arm, base_t)

    z_top = base_t[2, 3] - CUBE_SIZE / 2.0   # top face of the base cube

    # ── Remaining cubes ────────────────────────────────────────────────────────
    for i in range(1, n):
        t_robot_cube, _ = cube_results[i]
        print(f"\n── Cube {i+1}/{n} ──")

        grasp_cube(arm, t_robot_cube)

        target_z             = z_top - PLACE_MARGIN_M - CUBE_SIZE / 2.0
        stack_target         = base_t.copy()
        stack_target[2, 3]   = target_z
        stack_target[:3, :3] = base_t[:3, :3]   # keep base orientation

        place_cube(arm, stack_target)

        arm.set_position(z=22.5, relative=True, wait=True, speed=1000, mvacc=500)

        z_top = target_z - CUBE_SIZE / 2.0   # advance top face for next cube

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

        # Visualise detections
        for _, t_cam in cube_results:
            draw_pose_axes(cv_image, zed.camera_intrinsic, t_cam)
        cv2.imshow('Detected Cubes', cv_image)
        # if cv2.waitKey(0) != ord('k'):
        #     return

        stack_cubes(arm, cube_results)
        arm.move_gohome(wait=True, speed=600, mvacc=500)

    finally:
        arm.disconnect()
        zed.close()


if __name__ == "__main__":
    main()