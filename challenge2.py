"""
RRC Final Script: Size wise stacking

Detects all RGB cubes in the scene, estimates their size, sorts them largest to smallest, and stacks them into a single tower with the largest cube at the bottom and the smallest on top

Pipeline:
  1 Capture one synchronised RGB image + point cloud from the ZED camera
  2 Detect the four arena AprilTags to compute the camera ->robot transf
  3 For each colour (RGB), find all cube sized blobs via HSV masking and connected components
  4 For each blob, gather its 3D point cloud pts, transf them into the robot frame, slice the top 5 mm to isolate the top face, and fit a min atea rectangle to estimate the cube's XY centre, yaw, and side length
  5 Sort detected cubes largest first. The largest cube stays in place as the base, every other cube is picked and placed on top in order
  6 Show a visualisation window. Press 'k' to confirm and begin stacking, any other key to abort
"""

import cv2, numpy as np, time
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation
from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1challenge2 import grasp_cube_large, grasp_cube_small, place_cube, GRIPPER_LENGTH
import numpy 

ROBOT_IP = '192.168.1.182'

# Phy cube size bounds (metres)
# The rectangle footprint estimate is clamped to these limits to discard
# noise from incomplete point clouds or reflective surfaces
MIN_CUBE_SIZE_M = 0.015 # 15 mm: smallest cube in challenge
MAX_CUBE_SIZE_M = 0.040 # 40 mm: largest cube in challenge

STACK_BASE = [229.4, -296.4, 23.2]  # [x, y, z] in mm — adjust for your arena
STACK_BASE_M = np.array([
    STACK_BASE[0] / 1000.0,
    STACK_BASE[1] / 1000.0,
    STACK_BASE[2] / 1000.0,
])
SIZE_EPS = 0.003  # 3 mm tolerance for size-based decisions like grasping strategy or placement margins

# Stacking tuning
# Extra clearance added between the top face of the cube below and the bottom
# face of the cube being placed, to avoid a collision on descent
PLACE_MARGIN_M = 0.005 # 3 mm gap between stacked faces

# How far the arm retracts upward (mm) after each placement before moving
# to the next cube's position, so it doesn't collide with the growing tower
POST_PLACE_LIFT_MM = 40

PRE_GRASP_HEIGHT = 80

# Detection thresholds
MIN_COMPONENT_AREA = 500 # pixels: blobs smaller than this are ignored
MIN_3D_POINTS = 50  # min finite point cloud points per blob

# ── Tuning constants (replace the originals at the top) ─────────────────────
TOP_SURFACE_SLICE_M = 0.01   # 5 mm  — tight slice, excludes side faces
SIZE_EPS            = 0.0025  # 2.5 mm — separates 22.5 / 25 / 30 mm cleanly
#   22.5 mm → bucket round(22.5/2.5)=9
#   25.0 mm → bucket round(25.0/2.5)=10
#   30.0 mm → bucket round(30.0/2.5)=12

# HSV colour ranges
# Red wraps around the 0/180 hue boundary in OpenCV HSV, so it needs
# two separate ranges that are OR'd together (same as the cjeckpoints)
COLOUR_RANGES = [
    ('red', numpy.array([0,   80,  80]), numpy.array([10,  255, 255])),
    ('red', numpy.array([160, 80,  80]), numpy.array([180, 255, 255])),
    ('green', numpy.array([45,  80,  80]), numpy.array([75,  255, 255])),
    ('blue', numpy.array([95, 120, 100]), numpy.array([135, 255, 255])),
]

# Colour helpers

def to_hsv(image: numpy.ndarray) -> numpy.ndarray:
    """
    Convert a BGRA or BGR image to HSV
    The ZED camera outputs BGRA, so the alpha channel is stripped first if present before the colour space conv
    """
    if image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr = image
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

def is_top_face(comp_mask, point_cloud, t_cam_robot):
    """Check if a blob is a top face by computing its surface normal."""
    raw_points = point_cloud[comp_mask > 0][:, :3]
    cpoints = raw_points[numpy.all(numpy.isfinite(raw_points), axis=1)]
    
    if len(cpoints) < 20:
        return False
    
    # Transform to robot frame
    cpoints_m = cpoints / 1000.0
    T_cam_robot = numpy.linalg.inv(t_cam_robot)
    ones = numpy.ones((len(cpoints_m), 1))
    pts_robot = (T_cam_robot @ numpy.hstack([cpoints_m, ones]).T).T[:, :3]
    
    # Fit a plane using PCA — smallest eigenvector = surface normal
    centroid = pts_robot.mean(axis=0)
    centered = pts_robot - centroid
    cov = centered.T @ centered
    eigenvalues, eigenvectors = numpy.linalg.eigh(cov)
    normal = eigenvectors[:, 0]  # smallest eigenvalue = normal direction
    
    # Top face normal should be mostly vertical (large Z component)
    z_component = abs(normal[2])
    print(f"    Normal Z component: {z_component:.3f}")
    return z_component > 0.7  # threshold: 0.7 means within ~45° of vertical

def build_full_mask(hsv: numpy.ndarray, bgr: numpy.ndarray) -> numpy.ndarray:
    m1 = cv2.inRange(hsv, COLOUR_RANGES[0][1], COLOUR_RANGES[0][2])
    m2 = cv2.inRange(hsv, COLOUR_RANGES[1][1], COLOUR_RANGES[1][2])
    m3 = cv2.inRange(hsv, COLOUR_RANGES[2][1], COLOUR_RANGES[2][2])
    m4 = cv2.inRange(hsv, COLOUR_RANGES[3][1], COLOUR_RANGES[3][2])

    m4 = cv2.erode(m4, numpy.ones((3, 3), numpy.uint8), iterations=1)

    mask_red = cv2.bitwise_or(m1, m2)
    full_mask = cv2.bitwise_or(mask_red, m3)
    full_mask = cv2.bitwise_or(full_mask, m4)

    # Edge detection to separate top face from side faces
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    # Dilate edges slightly so they fully cut through the mask
    edges = cv2.dilate(edges, numpy.ones((3, 3), numpy.uint8), iterations=1)
    # Subtract edges from mask — this breaks top/side into separate blobs
    full_mask = cv2.bitwise_and(full_mask, cv2.bitwise_not(edges))

    # Clean up
    kernel = numpy.ones((5, 5), numpy.uint8)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('All Masks', full_mask)

    return full_mask

def dominant_colour(hsv: numpy.ndarray,
                    comp_mask: numpy.ndarray) -> str:
    """
    Returns the colour name with the most matching pixels inside comp_mask

    Each colour range is tested against only the pixels of this component because red has two ranges, their pixel counts are summed under the same 'red' key before taking the max across all colours
    """
    scores: dict[str, int] = {}
    for (name, lo, hi) in COLOUR_RANGES:
        hit = cv2.bitwise_and(cv2.inRange(hsv, lo, hi),
                              cv2.inRange(hsv, lo, hi),
                              mask=comp_mask)
        scores[name] = scores.get(name, 0) + int(numpy.count_nonzero(hit))
    return max(scores, key=scores.get) if scores else 'unknown'


def cube_sort_key(c):
    """
    Primary  : largest size bucket first  (30 mm before 25 mm before 22.5 mm)
    Secondary: left-to-right by robot-frame X within the same size class
               (adjust to `t_robot[1, 3]` if Y is your arena's left-to-right axis)
    Tertiary : raw size as a tie-breaker within the same X
    """
    size_bucket = round(c['size_m'] / SIZE_EPS)
    robot_x     = c['t_robot'][0, 3]   # ← swap for [1,3] if needed
    return (-size_bucket, robot_x, -c['size_m'])

# Every cube pose + size est.

def estimate_cube_pose(comp_mask, point_cloud, t_cam_robot, colour,
                       camera_intrinsic=None):
    """
    Estimates cube pose (position + yaw) and side length.

    Two independent size estimators are blended:

      A) 3-D minAreaRect  — fits a rectangle to the top-face XY footprint in
         the robot frame; robust when the point cloud is dense and the top face
         is mostly horizontal, but underestimates when the stereo cloud is sparse
         or the viewing angle leaves part of the top face occluded.

      B) Pixel-footprint + depth  — uses the pinhole model
             s = Z_cam × √(mask_px) / √(fx × fy)
         This is perspective-invariant: a cube twice as close subtends 4× more
         pixels but has half the depth, so the product stays constant.  It
         directly corrects the "closer looks bigger" artefact and is not
         affected by point-cloud sparsity.

    The two estimates are blended 40 % / 60 % favouring the pixel method.
    If camera_intrinsic is not supplied the function falls back to method A only.
    """
    T_robot_cam = t_cam_robot
    T_cam_robot = numpy.linalg.inv(T_robot_cam)

    # ── 1. Lighter erode — preserve edge points for accurate extent ──────────
    erode_kernel  = numpy.ones((2, 2), numpy.uint8)
    comp_mask_work = cv2.erode(comp_mask, erode_kernel, iterations=1)

    # ── 2. Collect valid 3-D points (camera frame, mm) ──────────────────────
    raw_points = point_cloud[comp_mask_work > 0][:, :3]
    cpoints    = raw_points[numpy.all(numpy.isfinite(raw_points), axis=1)]

    if len(cpoints) < MIN_3D_POINTS:
        return None

    cpoints_m = cpoints / 1000.0           # camera frame, metres

    # ── 3. Transform to robot frame ──────────────────────────────────────────
    ones          = numpy.ones((len(cpoints_m), 1))
    cpoints_robot = (T_cam_robot @ numpy.hstack([cpoints_m, ones]).T).T[:, :3]

    # ── 4. Tight top-surface slice (5 mm) ────────────────────────────────────
    z_max         = float(numpy.max(cpoints_robot[:, 2]))
    top_mask_bool = cpoints_robot[:, 2] > (z_max - TOP_SURFACE_SLICE_M)
    top_points    = cpoints_robot[top_mask_bool]       # robot frame
    top_cam_pts   = cpoints_m[top_mask_bool]           # camera frame (same indices)

    if len(top_points) < 10:
        return None

    # ── 5A. Method A — minAreaRect on robot-frame XY ─────────────────────────
    pts_2d = top_points[:, :2].astype(numpy.float32)
    rect   = cv2.minAreaRect(pts_2d)
    (center_xy, (w, h), angle_deg) = rect

    if w < h:
        angle_deg += 90.0
        w, h = h, w

    # Average of the two rectangle sides (more stable than hull sqrt for sparse clouds)
    size_3d = float((w + h) / 2.0)

    # ── 5B. Method B — pixel footprint + depth (perspective-invariant) ────────
    size_pixel = None
    if camera_intrinsic is not None:
        fx = float(camera_intrinsic[0, 0])
        fy = float(camera_intrinsic[1, 1])

        # Median depth of the top face in camera frame (metres)
        depth_cam = float(numpy.median(top_cam_pts[:, 2]))

        # Use the original (pre-erode) mask so we count every visible top pixel
        mask_px = int(numpy.count_nonzero(comp_mask))

        if mask_px > 0 and depth_cam > 0.01:
            # Pinhole: pixel_area = (s·fx/Z)·(s·fy/Z)  →  s = Z·√(px)·/√(fx·fy)
            size_pixel = depth_cam * numpy.sqrt(mask_px) / numpy.sqrt(fx * fy)

    # ── 6. Blend the two estimates ────────────────────────────────────────────
    if size_pixel is not None:
        raw_size = 0.40 * size_3d + 0.60 * size_pixel
    else:
        raw_size = size_3d

    estimated_size_m = float(numpy.clip(raw_size, MIN_CUBE_SIZE_M, MAX_CUBE_SIZE_M))

    pixel_str = f"{size_pixel*1000:.1f}" if size_pixel is not None else "N/A"
    depth_str = f"{top_cam_pts[:,2].mean()*1000:.0f}" if size_pixel is not None else "N/A"
    print(f"    [{colour}]  3D-rect={size_3d*1000:.1f} mm  "
          f"pixel={pixel_str} mm  "
          f"→ blended={estimated_size_m*1000:.1f} mm  "
          f"depth_cam={depth_str} mm")

    # ── 7. Cube centre Z = top-face Z minus half the estimated height ─────────
    center_z = z_max - estimated_size_m / 2.0

    # ── 8. Rotation: pure yaw, gripper Z pointing down ────────────────────────
    yaw_robot = numpy.deg2rad(angle_deg)
    Rz_robot  = numpy.array([
        [ numpy.cos(yaw_robot), -numpy.sin(yaw_robot), 0],
        [ numpy.sin(yaw_robot),  numpy.cos(yaw_robot), 0],
        [ 0,                     0,                    1],
    ]) @ numpy.diag([1, -1, -1])

    # ── 9. Assemble 4×4 transforms ────────────────────────────────────────────
    t_robot            = numpy.eye(4)
    t_robot[:3, :3]    = Rz_robot
    t_robot[:3,  3]    = [center_xy[0], center_xy[1], center_z]
    t_cam              = T_robot_cam @ t_robot

    return {
        'colour':  colour,
        'size_m':  estimated_size_m,
        't_robot': t_robot,
        't_cam':   t_cam,
    }

# Full-scene detection
def detect_all_cubes(image, point_cloud, t_cam_robot, camera_intrinsic=None):
    if image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr = image
    # ... rest stays the same
    hsv = to_hsv(image)
    full_mask = build_full_mask(hsv, bgr)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(full_mask)

    cubes = []
    for i in range(1, num_labels): # label 0 is the background
        if stats[i, cv2.CC_STAT_AREA] < MIN_COMPONENT_AREA:
            continue # too small, noise or reflection

        comp_mask = (labels == i).astype(numpy.uint8)

        # Skip side faces — only keep top faces
        if not is_top_face(comp_mask, point_cloud, t_cam_robot):
            print(f"  Skipping blob {i}: side face")
            continue

        colour = dominant_colour(hsv, comp_mask)
        result = estimate_cube_pose(comp_mask, point_cloud,
                                       t_cam_robot, colour, camera_intrinsic)
        if result is not None:
            cubes.append(result)
            print(f"  [{colour:5s}]  size ≈ {result['size_m']*1000:.1f} mm  |  "
                  f"robot XYZ = ({result['t_robot'][0,3]:.3f}, "
                  f"{result['t_robot'][1,3]:.3f}, "
                  f"{result['t_robot'][2,3]:.3f}) m")

    # Largest first: the biggest cube becomes the stable base of the tower
    cubes.sort(key=cube_sort_key)
    return cubes

# Stackin

def stack_cubes(arm, cubes: list[dict]) -> None:
    if not cubes:
        print("No cubes to stack.")
        return

    n = len(cubes)
    print(f"\nStacking order ({n} cubes, largest → smallest):")
    for i, c in enumerate(cubes):
        role = "BASE" if i == 0 else f"level {i + 1}"
        print(f"  {i+1}. {c['colour']:6s}  {c['size_m']*1000:.1f} mm  [{role}]")

    base_t = cubes[0]['t_robot'].copy()
    base_t[0, 3] = STACK_BASE[0] / 1000.0
    base_t[1, 3] = STACK_BASE[1] / 1000.0
    base_t[2, 3] = STACK_BASE[2] / 1000.0

    print(f"\n── Cube 1/{n}: {cubes[0]['colour']} (BASE) ──")
    if cubes[0]['size_m'] > 0.02:
        grasp_cube_large(arm, cubes[0]['t_robot'], cubes[0]['size_m'])
    else:
        grasp_cube_small(arm, cubes[0]['t_robot'], cubes[0]['size_m'])
    place_cube(arm, base_t, cubes[0]['size_m'])

    # Matches working checkpoint1challenge1 exactly — subtraction because
    # the xArm robot frame Z increases downward, so higher physical
    # positions have smaller (more negative) Z values
    z_top = base_t[2, 3] - cubes[0]['size_m'] / 2.0

    for i in range(1, n):
        cube = cubes[i]
        print(f"\n── Cube {i+1}/{n}: {cube['colour']}  ({cube['size_m']*1000:.1f} mm) ──")

        if cube['size_m'] > 0.02:
            grasp_cube_large(arm, cube['t_robot'], cube['size_m'])
        else:
            grasp_cube_small(arm, cube['t_robot'], cube['size_m'])

        target_z = z_top - PLACE_MARGIN_M - cube['size_m'] / 2.0

        stack_target = base_t.copy()
        stack_target[2, 3] = target_z
        stack_target[:3, :3] = base_t[:3, :3]

        place_cube(arm, stack_target, cube['size_m'])

        arm.set_position(z=POST_PLACE_LIFT_MM, relative=True, wait=True, speed=1000, mvacc=750)

        z_top = target_z - cube['size_m'] / 2.0

    print("\nAll cubes stacked.")

# Entry point

def main():
    zed = ZedCamera()
    arm = XArmAPI(ROBOT_IP)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)
    time.sleep(0.1)

    try:


        cv_image, point_cloud = zed.get_synchronized_frame()
        t_cam_robot = get_transform_camera_robot(cv_image, zed.camera_intrinsic)
        if t_cam_robot is None:
            return

        # Detect and size sort all coloured cubes in the scene
        print("\nDetecting cubes...")
        cubes = detect_all_cubes(cv_image, point_cloud, t_cam_robot, zed.camera_intrinsic)

        if not cubes:
            print("No cubes detected: exiting")
            return

        # Draw XYZ axes at each cube's estimated pose for visual verification
        for cube in cubes:
            draw_pose_axes(cv_image, zed.camera_intrinsic, cube['t_cam'])

        # Project each cube's 3D centre into 2D pixels and overlay a label
        # showing its stack rank, colour name, and estimated side length
        K = zed.camera_intrinsic
        for rank, cube in enumerate(cubes):
            pt = cube['t_cam'][:3, 3]
            if pt[2] > 0: # only project if point is in front of the cam
                u = int(K[0, 0] * pt[0] / pt[2] + K[0, 2])
                v = int(K[1, 1] * pt[1] / pt[2] + K[1, 2])
                label = f"#{rank+1} {cube['colour']} {cube['size_m']*1000:.0f}mm"
                cv2.putText(cv_image, label, (u, v - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 255), 2, cv2.LINE_AA)

        # Show the annotated image: press 'k' to confirm and begin stacking,
        # any other key to abort without moving the robot
        cv2.imshow('Detected Cubes: Size sorted stack', cv_image)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key != ord('k'):
            print("Aborted by user")
            return

        stack_cubes(arm, cubes)

        arm.move_gohome(wait=True, speed=1000, mvacc=750)

    finally:
        arm.disconnect()
        zed.close()


if __name__ == "__main__":
    main()