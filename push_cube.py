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
APPROACH_OFFSET = 0.07 # 7 cm behind cube — must clear gripper tip on descent
PUSH_DISTANCE = 0.06 # 6 cm forward push
PUSH_SPEED = 60 # mm/s — slow for controlled contact
TRAVEL_SPEED = 200 # mm/s — free-space moves


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


# ── Push Logic
def compute_push_direction(t_robot_cube, target_xy_mm):
    """
    Compute a unit push direction vector pointing from the cube toward
    the desired target position (in mm).

    If target_xy_mm is None, defaults to pushing along the robot +X axis.
    """
    if target_xy_mm is None:
        return numpy.array([1.0, 0.0])

    cube_xy = t_robot_cube[:2, 3] * 1000.0          # metres → mm
    target = numpy.array(target_xy_mm, dtype=float)
    vec = target - cube_xy
    norm = numpy.linalg.norm(vec)
    if norm < 1e-6:
        return numpy.array([1.0, 0.0])
    return vec / norm


def push_cube(arm, t_robot_cube, target_xy_mm=None,
              push_distance_m=PUSH_DISTANCE):
    """
    Push a single cube without grasping it

    Params
    arm: XArmAPI instance
    t_robot_cube: 4×4 pose of the cube in the robot frame (metres)
    target_xy_mm: [x, y] target in mm — the cube will be pushed toward this point.  Pass None to push along robot +X
    push_distance_m: how far (metres) the gripper travels through the cube
    """
    cube_pos = t_robot_cube[:3, 3] # metres
    dx, dy   = compute_push_direction(t_robot_cube, target_xy_mm)

    # Contact height: cube centre, nudged slightly below to avoid riding over
    contact_z_mm = (cube_pos[2] - CUBE_SIZE * 0.05) * 1000.0

    # Approach point: directly behind the cube along the push axis
    approach_x = (cube_pos[0] - dx * APPROACH_OFFSET) * 1000.0
    approach_y = (cube_pos[1] - dy * APPROACH_OFFSET) * 1000.0

    # End point: past the cube by push_distance_m
    end_x = (cube_pos[0] + dx * push_distance_m) * 1000.0
    end_y = (cube_pos[1] + dy * push_distance_m) * 1000.0

    print(f"  Cube centre  : ({cube_pos[0]*1000:.1f}, {cube_pos[1]*1000:.1f}) mm")
    print(f"  Push direction: ({dx:.3f}, {dy:.3f})")
    print(f"  Approach     : ({approach_x:.1f}, {approach_y:.1f}) mm")
    print(f"  End          : ({end_x:.1f}, {end_y:.1f}) mm")
    print(f"  Contact Z    : {contact_z_mm:.1f} mm")

    # Move above the approach point at safe height
    print("  [1/4] Moving above approach point...")
    arm.set_position(
        x=approach_x, y=approach_y, z=SAFE_Z_MM,
        roll=180, pitch=0, yaw=0,
        wait=True, speed=TRAVEL_SPEED, mvacc=500
    )

    # Desc to contact height
    print("  [2/4] Descending to contact height...")
    arm.set_position(
        z=contact_z_mm,
        wait=True, speed=100, mvacc=200
    )

    # Push straight through
    print("  [3/4] Pushing cube...")
    arm.set_position(
        x=end_x, y=end_y, z=contact_z_mm,
        wait=True, speed=PUSH_SPEED, mvacc=200
    )

    # Lift back to safe height
    print("  [4/4] Lifting clear...")
    arm.set_position(
        z=SAFE_Z_MM,
        wait=True, speed=TRAVEL_SPEED, mvacc=500
    )

    print("  Push complete.")

def main():
    zed = ZedCamera()
    arm = XArmAPI(ROBOT_IP)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)

    # Close the gripper so it acts as a solid pusher
    arm.set_gripper_position(0, wait=True)

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

        print(f"Detected {len(cube_results)} cube(s). Using the first one.")

        # Visualise detections
        for _, t_cam in cube_results:
            draw_pose_axes(cv_image, zed.camera_intrinsic, t_cam)
        cv2.imshow('Detected Cubes', cv_image)
        if cv2.waitKey(0) != ord('k'):
            print("Aborted by user.")
            return

        # Push the first detected cube
        # Edit target_xy_mm to push toward a specific location,
        # or set to None to push along +X.
        target_xy_mm = None  # change to desired target

        t_robot_cube, _ = cube_results[0]
        print("\nPushing cube...")
        push_cube(arm, t_robot_cube, target_xy_mm=target_xy_mm)

        arm.move_gohome(wait=True, speed=TRAVEL_SPEED, mvacc=500)

    finally:
        arm.disconnect()
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()