"""
RRC Final Script: Size-wise stacking

Detects all RGB cubes in the scene, estimates their size, sorts them largest
to smallest, and stacks them into a single tower with the largest cube at the
bottom and the smallest on top.

Pipeline:
  1 Capture one synchronised RGB image + point cloud from the ZED camera
  2 Detect the four arena AprilTags to compute the camera -> robot transform
  3 For each colour (RGB), find all cube-sized blobs via HSV masking and
    connected components
  4 For each blob, gather its 3D point cloud pts, transform them into the
    robot frame, slice the top 5 mm to isolate the top face, and fit a
    min-area rectangle to estimate the cube's XY centre, yaw, and side length
  5 Sort detected cubes largest first. The largest cube stays in place as the
    base; every other cube is picked and placed on top in order
  6 Show a visualisation window. Press 'k' to confirm and begin stacking,
    any other key to abort
"""

import cv2, numpy as np, time
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation
from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1challenge2 import grasp_cube_large, grasp_cube_small, place_cube, GRIPPER_LENGTH
import numpy

ROBOT_IP = '192.168.1.183'

# Physical cube size bounds (metres)
MIN_CUBE_SIZE_M = 0.015   # 15 mm: smallest cube in challenge
MAX_CUBE_SIZE_M = 0.040   # 40 mm: largest cube in challenge

STACK_BASE   = [229.4, -296.4, 23.2]   # [x, y, z] in mm — adjust for your arena
STACK_BASE_M = np.array([
    STACK_BASE[0] / 1000.0,
    STACK_BASE[1] / 1000.0,
    STACK_BASE[2] / 1000.0,
])
SIZE_EPS = 0.003   # 3 mm tolerance for size-based decisions

# Stacking tuning
PLACE_MARGIN_M = 0.008   # 8 mm gap between stacked faces

# Detection thresholds
MIN_COMPONENT_AREA   = 500    # pixels: blobs smaller than this are ignored
MIN_3D_POINTS        = 50    # min finite point cloud points per blob
TOP_SURFACE_SLICE_M  = 0.01  # 10 mm: height of the top surface slice

# HSV colour ranges
COLOUR_RANGES = [
    ('red',   numpy.array([0,   80,  80]), numpy.array([10,  255, 255])),
    ('red',   numpy.array([160, 80,  80]), numpy.array([180, 255, 255])),
    ('green', numpy.array([45,  80,  80]), numpy.array([75,  255, 255])),
    ('blue',  numpy.array([95,  80,  80]), numpy.array([135, 255, 255])),
]


# ── Colour helpers ────────────────────────────────────────────────────────────

def to_hsv(image: numpy.ndarray) -> numpy.ndarray:
    """Convert a BGRA or BGR image to HSV."""
    bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) if image.shape[2] == 4 else image
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def is_top_face(comp_mask, point_cloud, t_cam_robot):
    """Check if a blob is a top face by computing its surface normal via PCA."""
    raw_points = point_cloud[comp_mask > 0][:, :3]
    cpoints    = raw_points[numpy.all(numpy.isfinite(raw_points), axis=1)]

    if len(cpoints) < 20:
        return False

    cpoints_m  = cpoints / 1000.0
    T_cam_robot = numpy.linalg.inv(t_cam_robot)
    ones        = numpy.ones((len(cpoints_m), 1))
    pts_robot   = (T_cam_robot @ numpy.hstack([cpoints_m, ones]).T).T[:, :3]

    centroid   = pts_robot.mean(axis=0)
    centered   = pts_robot - centroid
    cov        = centered.T @ centered
    eigenvalues, eigenvectors = numpy.linalg.eigh(cov)
    normal     = eigenvectors[:, 0]   # smallest eigenvalue → normal direction

    z_component = abs(normal[2])
    print(f"    Normal Z component: {z_component:.3f}")
    return z_component > 0.7   # within ~45° of vertical


def build_full_mask(hsv: numpy.ndarray, bgr: numpy.ndarray) -> numpy.ndarray:
    m1 = cv2.inRange(hsv, COLOUR_RANGES[0][1], COLOUR_RANGES[0][2])
    m2 = cv2.inRange(hsv, COLOUR_RANGES[1][1], COLOUR_RANGES[1][2])
    m3 = cv2.inRange(hsv, COLOUR_RANGES[2][1], COLOUR_RANGES[2][2])
    m4 = cv2.inRange(hsv, COLOUR_RANGES[3][1], COLOUR_RANGES[3][2])

    mask_red  = cv2.bitwise_or(m1, m2)
    full_mask = cv2.bitwise_or(mask_red, m3)
    full_mask = cv2.bitwise_or(full_mask, m4)

    # Canny edges break top/side faces into separate blobs
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    edges = cv2.dilate(edges, numpy.ones((3, 3), numpy.uint8), iterations=1)
    full_mask = cv2.bitwise_and(full_mask, cv2.bitwise_not(edges))

    kernel    = numpy.ones((5, 5), numpy.uint8)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN,  kernel)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('All Masks', full_mask)
    return full_mask


def dominant_colour(hsv: numpy.ndarray, comp_mask: numpy.ndarray) -> str:
    """Return the colour name with the most matching pixels inside comp_mask."""
    scores: dict[str, int] = {}
    for (name, lo, hi) in COLOUR_RANGES:
        hit = cv2.bitwise_and(cv2.inRange(hsv, lo, hi),
                              cv2.inRange(hsv, lo, hi),
                              mask=comp_mask)
        scores[name] = scores.get(name, 0) + int(numpy.count_nonzero(hit))
    return max(scores, key=scores.get) if scores else 'unknown'


def cube_sort_key(c):
    size = c['size_m']
    dx   = c['t_robot'][0, 3] - STACK_BASE_M[0]
    dy   = c['t_robot'][1, 3] - STACK_BASE_M[1]
    dist = np.sqrt(dx**2 + dy**2)
    size_bucket = round(size / SIZE_EPS)
    return (-size_bucket, dist, -size)


# ── Pose estimation ───────────────────────────────────────────────────────────

def estimate_cube_pose(comp_mask, point_cloud, t_cam_robot, colour) -> dict | None:
    """
    Estimate a cube's pose (position + yaw) and side length from its 2D
    connected-component mask and the corresponding point cloud.
    Returns a dict with keys: colour, size_m, t_robot, t_cam — or None.
    """
    T_robot_cam = t_cam_robot
    T_cam_robot = numpy.linalg.inv(T_robot_cam)

    comp_mask  = cv2.erode(comp_mask, numpy.ones((3, 3), numpy.uint8), iterations=1)
    raw_points = point_cloud[comp_mask > 0][:, :3]
    cpoints    = raw_points[numpy.all(numpy.isfinite(raw_points), axis=1)]

    if len(cpoints) < MIN_3D_POINTS:
        return None

    cpoints_m    = cpoints / 1000.0
    ones         = numpy.ones((len(cpoints_m), 1))
    cpoints_robot = (T_cam_robot @ numpy.hstack([cpoints_m, ones]).T).T[:, :3]

    z_max      = float(numpy.max(cpoints_robot[:, 2]))
    top_mask   = cpoints_robot[:, 2] > (z_max - TOP_SURFACE_SLICE_M)
    top_points = cpoints_robot[top_mask]

    if len(top_points) < 10:
        return None

    pts_2d_top = top_points[:, :2].astype(numpy.float32)
    rect       = cv2.minAreaRect(pts_2d_top)
    (center_xy, (w, h), angle_deg) = rect

    if w < h:
        angle_deg += 90.0
        w, h = h, w

    # Size from convex hull area — more robust than edge lengths
    hull           = cv2.convexHull(pts_2d_top)
    hull_area      = cv2.contourArea(hull)
    estimated_size = float(numpy.clip(numpy.sqrt(hull_area),
                                      MIN_CUBE_SIZE_M, MAX_CUBE_SIZE_M))
    print(f"    minAreaRect: w={w*1000:.1f}mm h={h*1000:.1f}mm | "
          f"hull_area size={estimated_size*1000:.1f}mm")

    center_z  = z_max - (estimated_size / 2.0)
    yaw_robot = numpy.deg2rad(angle_deg)
    Rz_robot  = numpy.array([
        [numpy.cos(yaw_robot), -numpy.sin(yaw_robot), 0],
        [numpy.sin(yaw_robot),  numpy.cos(yaw_robot), 0],
        [0,                     0,                    1],
    ]) @ numpy.diag([1, -1, -1])

    t_robot          = numpy.eye(4)
    t_robot[:3, :3]  = Rz_robot
    t_robot[:3,  3]  = [center_xy[0], center_xy[1], center_z]
    t_cam            = T_robot_cam @ t_robot

    return {'colour': colour, 'size_m': estimated_size,
            't_robot': t_robot, 't_cam': t_cam}


# ── Full-scene detection ──────────────────────────────────────────────────────

def detect_all_cubes(image, point_cloud, t_cam_robot):
    bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) if image.shape[2] == 4 else image
    hsv = to_hsv(image)
    full_mask = build_full_mask(hsv, bgr)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(full_mask)

    cubes = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < MIN_COMPONENT_AREA:
            continue

        comp_mask = (labels == i).astype(numpy.uint8)

        if not is_top_face(comp_mask, point_cloud, t_cam_robot):
            print(f"  Skipping blob {i}: side face")
            continue

        colour = dominant_colour(hsv, comp_mask)
        result = estimate_cube_pose(comp_mask, point_cloud, t_cam_robot, colour)
        if result is not None:
            cubes.append(result)
            print(f"  [{colour:5s}]  size ≈ {result['size_m']*1000:.1f} mm  |  "
                  f"robot XYZ = ({result['t_robot'][0,3]:.3f}, "
                  f"{result['t_robot'][1,3]:.3f}, "
                  f"{result['t_robot'][2,3]:.3f}) m")

    cubes.sort(key=cube_sort_key)
    return cubes


# ── Stacking ──────────────────────────────────────────────────────────────────

def stack_cubes(arm, cubes: list[dict]) -> None:
    """
    Stack all detected cubes largest → smallest.

    The largest cube (index 0) is moved to STACK_BASE and becomes the base.
    All subsequent cubes are placed on top with a dynamically computed target Z.

    No explicit post-place lift: grasp_cube_* opens the gripper and moves to
    PRE_GRASP_HEIGHT as its very first action, which already clears the stack.
    """
    if not cubes:
        print("No cubes to stack.")
        return

    n = len(cubes)
    print(f"\nStacking order ({n} cubes, largest → smallest):")
    for i, c in enumerate(cubes):
        role = "BASE" if i == 0 else f"level {i + 1}"
        print(f"  {i+1}. {c['colour']:6s}  {c['size_m']*1000:.1f} mm  [{role}]")

    base_t          = cubes[0]['t_robot'].copy()
    base_t[0, 3]    = STACK_BASE[0] / 1000.0
    base_t[1, 3]    = STACK_BASE[1] / 1000.0
    base_t[2, 3]    = STACK_BASE[2] / 1000.0

    print(f"\n── Cube 1/{n}: {cubes[0]['colour']} (BASE) ──")
    if cubes[0]['size_m'] > 0.02:
        grasp_cube_large(arm, cubes[0]['t_robot'], cubes[0]['size_m'])
    else:
        grasp_cube_small(arm, cubes[0]['t_robot'], cubes[0]['size_m'])
    place_cube(arm, base_t, cubes[0]['size_m'])

    z_top = base_t[2, 3] - cubes[0]['size_m'] / 2.0

    for i in range(1, n):
        cube = cubes[i]
        print(f"\n── Cube {i+1}/{n}: {cube['colour']}  ({cube['size_m']*1000:.1f} mm) ──")

        if cube['size_m'] > 0.02:
            grasp_cube_large(arm, cube['t_robot'], cube['size_m'])
        else:
            grasp_cube_small(arm, cube['t_robot'], cube['size_m'])

        target_z             = z_top - PLACE_MARGIN_M - cube['size_m'] / 2.0
        stack_target         = base_t.copy()
        stack_target[2, 3]   = target_z
        stack_target[:3, :3] = base_t[:3, :3]

        place_cube(arm, stack_target, cube['size_m'])

        # POST-PLACE LIFT REMOVED — grasp_cube_* moves to PRE_GRASP_HEIGHT
        # as its first action, which already clears the growing stack.
        # Removing this saves ~0.5 s × (n-1) placements.

        z_top = target_z - cube['size_m'] / 2.0

    print("\nAll cubes stacked.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    zed = ZedCamera()
    arm = XArmAPI(ROBOT_IP)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)
    # time.sleep(0.1) removed — move_gohome(wait=True) already blocks until complete

    try:
        cv_image, point_cloud = zed.get_synchronized_frame()
        t_cam_robot = get_transform_camera_robot(cv_image, zed.camera_intrinsic)
        if t_cam_robot is None:
            return

        print("\nDetecting cubes...")
        cubes = detect_all_cubes(cv_image, point_cloud, t_cam_robot)

        if not cubes:
            print("No cubes detected: exiting")
            return

        for cube in cubes:
            draw_pose_axes(cv_image, zed.camera_intrinsic, cube['t_cam'])

        K = zed.camera_intrinsic
        for rank, cube in enumerate(cubes):
            pt = cube['t_cam'][:3, 3]
            if pt[2] > 0:
                u = int(K[0, 0] * pt[0] / pt[2] + K[0, 2])
                v = int(K[1, 1] * pt[1] / pt[2] + K[1, 2])
                label = f"#{rank+1} {cube['colour']} {cube['size_m']*1000:.0f}mm"
                cv2.putText(cv_image, label, (u, v - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Detected Cubes: Size sorted stack', cv_image)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key != ord('k'):
            print("Aborted by user")
            return

        stack_cubes(arm, cubes)
        arm.move_gohome(wait=True)

    finally:
        arm.disconnect()
        zed.close()


if __name__ == "__main__":
    main()