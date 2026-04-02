"""
RRC Final Script: Size wise stacking

Detects all RGB cubes in the scene, estimates their size, sorts them largest to smallest, and stacks them into a single tower with the largest cube at the bottom and the smallest on top

Pipeline:
  1 Capture one synchronised RGB image + point cloud from the ZED camera
  2 Detect the four arena AprilTags to compute the camera ->robot transf
  3 For each colour (RGB), find all cube sized blobs via HSV masking and connected components
  4 For each blob, gather its 3D point cloud pts, transf them into the robot frame, slice the top 5 mm to isolate the top face, and fit a min atea rectangle to estimate the cube's XY centre, yaw, and side length
  5 Try every permutation of cubes (index 0 = immovable base).
    For each permutation compute total XY arm travel:
      home->pick[1]->place(base)->pick[2]->place(base)->...->home
    Select the permutation with the shortest total distance.
  6 Show a visualisation window. Press 'k' to confirm and begin stacking, any other key to abort
"""

import cv2, numpy as np, time, itertools
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation
from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1challenge2 import grasp_cube_large, grasp_cube_small, place_cube, GRIPPER_LENGTH
import numpy 

ROBOT_IP = '192.168.1.183'

# Phy cube size bounds (metres)
# The rectangle footprint estimate is clamped to these limits to discard
# noise from incomplete point clouds or reflective surfaces
MIN_CUBE_SIZE_M = 0.015 # 15 mm: smallest cube in challenge
MAX_CUBE_SIZE_M = 0.040 # 40 mm: largest cube in challenge

SIZE_EPS = 0.001  # 3 mm tolerance for size-based decisions like grasping strategy or placement margins

# Stacking tuning
# Extra clearance added between the top face of the cube below and the bottom
# face of the cube being placed, to avoid a collision on descent
PLACE_MARGIN_M = 0.008 # 3 mm gap between stacked faces

# How far the arm retracts upward (mm) after each placement before moving
# to the next cube's position, so it doesn't collide with the growing tower
POST_PLACE_LIFT_MM = 20

PRE_GRASP_HEIGHT = 40

# Detection thresholds
MIN_COMPONENT_AREA = 500 # pixels: blobs smaller than this are ignored
MIN_3D_POINTS = 50  # min finite point cloud points per blob
TOP_SURFACE_SLICE_M = 0.01 # 5 mm: height of the top surface slice used
                            # for centre and orient est.

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



# Home position in robot-frame metres (arm always starts and ends here)
HOME_XY_M = numpy.array([0.0, 0.0])


def _xy(cube: dict) -> numpy.ndarray:
    """Return the XY position of a cube in robot-frame metres."""
    t = cube['t_robot']
    return numpy.array([t[0, 3], t[1, 3]])


def _travel_cost(perm: list) -> float:
    """
    Compute total XY arm travel for a given cube ordering.

    perm[0] is the base (not moved).  The arm visits every other cube in
    order:  home -> pick perm[1] -> place at base -> pick perm[2] -> ... -> home

    Only XY distance is used because Z travel (PRE_GRASP_HEIGHT) is
    identical for every move and cancels out when comparing permutations.
    """
    if len(perm) == 1:
        return 0.0  # single cube: base only, nothing to move

    base_xy = _xy(perm[0])
    pos     = HOME_XY_M
    total   = 0.0

    for cube in perm[1:]:
        pick_xy = _xy(cube)
        total  += numpy.linalg.norm(pick_xy - pos)       # travel to pick
        total  += numpy.linalg.norm(base_xy - pick_xy)   # travel to place
        pos     = base_xy

    total += numpy.linalg.norm(HOME_XY_M - pos)          # return home
    return total


def fastest_order(cubes: list) -> list:
    """
    Try every permutation of cubes and return the ordering (base first)
    that minimises total XY arm travel distance.

    Feasible for the small cube counts in the challenge
    (<=6 cubes -> <=720 permutations evaluated in microseconds).
    """
    if len(cubes) <= 1:
        return cubes

    best_cost = float("inf")
    best_perm = cubes

    for perm in itertools.permutations(cubes):
        cost = _travel_cost(list(perm))
        if cost < best_cost:
            best_cost = cost
            best_perm = list(perm)

    n_perms = 1
    for k in range(1, len(cubes) + 1):
        n_perms *= k
    print(f"Optimal travel distance: {best_cost*1000:.0f} mm  "
          f"(evaluated {n_perms} permutations)")
    return best_perm


# Every cube pose + size est.

def estimate_cube_pose(comp_mask:   numpy.ndarray, point_cloud: numpy.ndarray, t_cam_robot: numpy.ndarray, colour: str) -> dict | None:
    """
    Est a cube's pose (pos + yaw) and side length from its 2D connected component mask and the corresponding point cloud

    Returns a dict with keys: colour, size_m, t_robot (4×4), t_cam (4×4)
    Returns None if there are too few valid 3D points to proceed

    Process:
      1 Erode the mask by 1 pixel to remove noisy edge measurements where the stereo depth is least reliable
      2 Index the point cloud with the eroded mask to get the cube's 3D points in camera frame (mm)
      3 Filter out any NaN / Inf values produced by the stereo estimator on textureless or hazy regions
      4 Convert mm to metres, append a homogeneous 1 to each point, and multiply by inv(t_cam_robot) to bring them into the robot frame
      5 Find z_max: the highest Z value, which is the top face of the cube
      6 Keep only points within TOP_SURFACE_SLICE_M of z_max to isolate the flat top face and discard the sides
      7 Fit a min area rectangle to the XY coordinates of those points. The rectangle centre gives the cube's XY position, its rot angle gives the yaw, avg the two side lengths estimates size
      8 Clamp the size estimate to MIN_CUBE_SIZE_M and MAX_CUBE_SIZE_M to reject outliers caused by sparse or noisy point clouds
      9 Set center_z = z_max − size/2 to place the pose at the geometric centre of the cube rather than its top face
     10 Build a rot matrix: a pure yaw Rz multiplied by diag(1,−1,−1) so the gripper's Z axis points downward into the top face
     11 Assemble t_robot (pose in robot fr) and t_cam (pose back in cam fr), the latter needed by draw_pose_axes for overlay
    """
    # inv(t_cam_robot) converts homg cam-fr points -> robot fr:
    # p_robot = inv(t_cam_robot) @ p_cam_homogeneous
    T_robot_cam = t_cam_robot # t_cam_robot as received
    T_cam_robot = numpy.linalg.inv(T_robot_cam) # used to transform points

    # Step 1: erode to strip unreliable edge pixels
    comp_mask = cv2.erode(comp_mask, numpy.ones((3, 3), numpy.uint8), iterations=1)

    # Steps 2 and 3: gather finite 3D points under this component
    raw_points = point_cloud[comp_mask > 0][:, :3]
    cpoints = raw_points[numpy.all(numpy.isfinite(raw_points), axis=1)]

    if len(cpoints) < MIN_3D_POINTS:
        return None

    # Step 4: mm to metrez, then transform into robot frame
    cpoints_m = cpoints / 1000.0
    ones = numpy.ones((len(cpoints_m), 1))
    cpoints_robot = (T_cam_robot @ numpy.hstack([cpoints_m, ones]).T).T[:, :3]

    # Steps 5 and 6: isolate the top surface slice
    z_max = float(numpy.max(cpoints_robot[:, 2]))
    top_mask = cpoints_robot[:, 2] > (z_max - TOP_SURFACE_SLICE_M)
    top_points = cpoints_robot[top_mask]

    if len(top_points) < 10:
        return None

    # Step 7: fit min area rectangle to top surface XY footprint
    pts_2d_top = top_points[:, :2].astype(numpy.float32)
    rect = cv2.minAreaRect(pts_2d_top)
    (center_xy, (w, h), angle_deg) = rect

    # Ensure w is always the longer side so angle_deg is consistent
    if w < h:
        angle_deg += 90.0
        w, h = h, w

    # Step 8: estimate size from AREA — much more robust than edge lengths
    # For a square top face: area = side^2, so side = sqrt(area)
    # Use convex hull area to ignore internal holes/noise
    hull = cv2.convexHull(pts_2d_top)
    hull_area = cv2.contourArea(hull)
    estimated_size_m = float(numpy.clip(
        numpy.sqrt(hull_area),
        MIN_CUBE_SIZE_M,
        MAX_CUBE_SIZE_M
    ))
    print(f"    minAreaRect: w={w*1000:.1f}mm h={h*1000:.1f}mm | hull_area size={estimated_size_m*1000:.1f}mm")

    # Step 9: cube centre Z sits half a cube height below the top face
    center_z = z_max - (estimated_size_m / 2.0)

    # Step 10: rot - yaw only (cube is upright), then flip Y and Z so the gripper Z axis points downward ont o the top face
    yaw_robot = numpy.deg2rad(angle_deg)
    Rz_robot  = numpy.array([
        [numpy.cos(yaw_robot), -numpy.sin(yaw_robot), 0],
        [numpy.sin(yaw_robot),  numpy.cos(yaw_robot), 0],
        [0,                     0,                    1],
    ]) @ numpy.diag([1, -1, -1])

    # Step 11: assemble 4×4 transforms
    t_robot = numpy.eye(4)
    t_robot[:3, :3] = Rz_robot
    t_robot[:3,  3] = [center_xy[0], center_xy[1], center_z]

    # t_cam expresses the cube pose in camera frame so draw_pose_axes can
    # project the axes onto the image using the camera intrinsics
    t_cam = T_robot_cam @ t_robot

    return {
        'colour':  colour,
        'size_m':  estimated_size_m,
        't_robot': t_robot,
        't_cam':   t_cam,
    }

# Full-scene detection
def detect_all_cubes(image, point_cloud, t_cam_robot):
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
                                       t_cam_robot, colour)
        if result is not None:
            cubes.append(result)
            print(f"  [{colour:5s}]  size ≈ {result['size_m']*1000:.1f} mm  |  "
                  f"robot XYZ = ({result['t_robot'][0,3]:.3f}, "
                  f"{result['t_robot'][1,3]:.3f}, "
                  f"{result['t_robot'][2,3]:.3f}) m")

    # Order cubes for minimum total arm travel (base = index 0, never moved)
    cubes = fastest_order(cubes)
    return cubes

# Stackin

def stack_cubes(arm, cubes: list[dict]) -> None:
    """
    Stack all detected cubes in the pre-computed fastest-travel order.

    cubes[0] is the immovable base (chosen to minimise total arm travel).
    All subsequent cubes are picked from their detected positions and
    placed on top of the growing tower in the order that minimises total
    XY arm travel.

    Target Z for each placement:
      target_z = z_top - PLACE_MARGIN_M - (this cube half-height)
    After placing, z_top advances so the next cube lands correctly.
    The XY position and yaw are always copied from the base cube's
    transform, so all cubes land on the same XY footprint.
    """
    if not cubes:
        print("No cubes to stack.")
        return

    n = len(cubes)
    print(f"\nStacking order ({n} cubes, fastest-travel order):")
    for i, c in enumerate(cubes):
        role = "BASE" if i == 0 else f"level {i + 1}"
        print(f"  {i+1}. {c['colour']:6s}  {c['size_m']*1000:.1f} mm  [{role}]")

    # The base cube stays exactly where it is — use its detected pose as the XY/yaw reference
    base_t = cubes[0]['t_robot'].copy()

    # z_top starts at the top face of the base cube (centre_z - half-height)
    z_top = base_t[2, 3] - cubes[0]['size_m'] / 2.0

    print(f"\n── Cube 1/{n}: {cubes[0]['colour']} (BASE — not moved) ──")

    for i in range(1, n):
        cube = cubes[i]
        print(f"\n── Cube {i+1}/{n}: {cube['colour']}  "
              f"({cube['size_m']*1000:.1f} mm) ──")

        if cube['size_m'] > 0.02:
            grasp_cube_large(arm, cube['t_robot'], cube['size_m'])
        else:
            grasp_cube_small(arm, cube['t_robot'], cube['size_m'])

        target_z = z_top - PLACE_MARGIN_M - cube['size_m'] / 2.0

        # stack_target = base_t.copy()
        # stack_target[2, 3] = target_z

        # # extract position — convert metres to mm for xArm API
        # x = stack_target[0, 3] * 1000
        # y = stack_target[1, 3] * 1000
        # z = stack_target[2, 3] * 1000
    
        # # extract yaw from rotation matrix
        # rot = Rotation.from_matrix(stack_target[:3, :3])
        # r, p, yaw = rot.as_euler('xyz', degrees=False)
    
        # # move to pre-place height above target
        # arm.set_position(x, y, -z + PRE_GRASP_HEIGHT, r, p, yaw, is_radian=True, wait=True)
        # time.sleep(0.5)

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
        cubes = detect_all_cubes(cv_image, point_cloud, t_cam_robot)

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
        cv2.imshow('Detected Cubes: Fastest-travel stack order', cv_image)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key != ord('k'):
            print("Aborted by user")
            return

        stack_cubes(arm, cubes)

        arm.move_gohome(wait=True, speed = 1000, mvacc=700)

    finally:
        arm.disconnect()
        zed.close()


if __name__ == "__main__":
    main()