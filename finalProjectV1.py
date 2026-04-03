import cv2, numpy as np, time
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1challenge2 import grasp_cube_large, grasp_cube_small, place_cube, GRIPPER_LENGTH

from voxel import voxelize_between_three_tags

ROBOT_IP = '192.168.1.182'

MIN_COMPONENT_AREA = 500
MIN_3D_POINTS = 50
TOP_SURFACE_SLICE_M = 0.01

MIN_BLOCK_SIZE_M = 0.023
MAX_BLOCK_SIZE_M = 0.080

PREVIEW = True

# -------------------------------------------------
# USER-PROVIDED PLACEMENT SPECS
# -------------------------------------------------

# Red blocks: first drop location, then +27 mm in X for each next red block

RED_START_X_MM = 100
RED_START_Y_MM = 200
RED_PLACE_Z_MM = 60
RED_STEP_X_MM  = 35

# Red placement orientation
RED_R_DEG = -180
RED_P_DEG = 0
RED_YAW_DEG = -90

# Green placement target:
# Fill X/Y once the setup photo is available or if you want this tied

GREEN_PLACE_X_MM = 100   
GREEN_PLACE_Y_MM = -240     
GREEN_PLACE_Z_MM = 60

GREEN_R_DEG = -180
GREEN_P_DEG = 0
GREEN_YAW_DEG = -90

COLOUR_RANGES = [
    ('red',   np.array([0,   80,  80]), np.array([10,  255, 255])),
    ('red',   np.array([160, 80,  80]), np.array([180, 255, 255])),
    ('green', np.array([45,  80,  80]), np.array([75,  255, 255])),
    ('blue',  np.array([95, 120, 100]), np.array([135, 255, 255])),
]

# -------------------------------------------------
# Image helpers
# -------------------------------------------------

def to_hsv(image: np.ndarray) -> np.ndarray:
    if image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr = image
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

def build_full_mask(hsv: np.ndarray, bgr: np.ndarray) -> np.ndarray:
    masks = []
    for _, lo, hi in COLOUR_RANGES:
        masks.append(cv2.inRange(hsv, lo, hi))

    full_mask = masks[0]
    for m in masks[1:]:
        full_mask = cv2.bitwise_or(full_mask, m)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    full_mask = cv2.bitwise_and(full_mask, cv2.bitwise_not(edges))

    kernel = np.ones((5, 5), np.uint8)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)
    
    cv2.imshow('All Masks', full_mask)
    
    return full_mask

def dominant_colour(hsv: np.ndarray, comp_mask: np.ndarray) -> str:
    scores = {}
    for (name, lo, hi) in COLOUR_RANGES:
        hit = cv2.bitwise_and(cv2.inRange(hsv, lo, hi),
                              cv2.inRange(hsv, lo, hi),
                              mask=comp_mask)
        scores[name] = scores.get(name, 0) + int(np.count_nonzero(hit))
    return max(scores, key=scores.get) if scores else 'unknown'

def is_top_face(comp_mask, point_cloud, t_cam_robot):
    raw_points = point_cloud[comp_mask > 0][:, :3]
    cpoints = raw_points[np.all(np.isfinite(raw_points), axis=1)]
    if len(cpoints) < 20:
        return False

    cpoints_m = cpoints / 1000.0
    T_cam_robot = np.linalg.inv(t_cam_robot)
    ones = np.ones((len(cpoints_m), 1))
    pts_robot = (T_cam_robot @ np.hstack([cpoints_m, ones]).T).T[:, :3]

    centroid = pts_robot.mean(axis=0)
    centered = pts_robot - centroid
    cov = centered.T @ centered
    _, eigenvectors = np.linalg.eigh(cov)
    normal = eigenvectors[:, 0]
    return abs(normal[2]) > 0.7

# -------------------------------------------------
# Pose estimation
# -------------------------------------------------

def green_top_face_fully_visible(block, image_shape):
    u, v = block['image_center_px']
    H, W = image_shape[:2]

    border_margin_px = 40
    if u < border_margin_px or u > (W - border_margin_px):
        print("green rejected: too close to horizontal border")
        return False
    if v < border_margin_px or v > (H - border_margin_px):
        print("green rejected: too close to vertical border")
        return False

    print(f"green stats: square_ratio={block['square_ratio']:.3f}, "
          f"rect_w={block['rect_w_m']:.3f}, rect_h={block['rect_h_m']:.3f}, "
          f"size={block['size_m']:.3f}")

    if block['square_ratio'] < 0.35:
        print("green rejected: square_ratio too low")
        return False
    if not (0.009 <= block['rect_w_m'] <= 0.040):
        print("green rejected: rect_w out of range")
        return False
    if not (0.009 <= block['rect_h_m'] <= 0.040):
        print("green rejected: rect_h out of range")
        return False
    if not (0.009 <= block['size_m'] <= 0.040):
        print("green rejected: size out of range")
        return False

    print("green accepted")
    return True

def estimate_block_pose(comp_mask, point_cloud, t_cam_robot, colour):
    T_robot_cam = t_cam_robot
    T_cam_robot = np.linalg.inv(T_robot_cam)

    comp_mask = cv2.erode(comp_mask, np.ones((3, 3), np.uint8), iterations=1)

    raw_points = point_cloud[comp_mask > 0][:, :3]
    cpoints = raw_points[np.all(np.isfinite(raw_points), axis=1)]
    if len(cpoints) < MIN_3D_POINTS:
        return None

    cpoints_m = cpoints / 1000.0
    ones = np.ones((len(cpoints_m), 1))
    cpoints_robot = (T_cam_robot @ np.hstack([cpoints_m, ones]).T).T[:, :3]

    z_max = float(np.max(cpoints_robot[:, 2]))
    top_mask = cpoints_robot[:, 2] > (z_max - TOP_SURFACE_SLICE_M)
    top_points = cpoints_robot[top_mask]
    if len(top_points) < 10:
        return None

    pts_2d_top = top_points[:, :2].astype(np.float32)
    rect = cv2.minAreaRect(pts_2d_top)
    (center_xy, (w, h), angle_deg) = rect

    if w < h:
        angle_deg += 90.0
        w, h = h, w

    hull = cv2.convexHull(pts_2d_top)
    hull_area = cv2.contourArea(hull)
    estimated_size_m = float(np.clip(np.sqrt(max(hull_area, 1e-8)),
                                     MIN_BLOCK_SIZE_M,
                                     MAX_BLOCK_SIZE_M))

    center_z = z_max - (estimated_size_m / 2.0)

    yaw_robot = np.deg2rad(angle_deg)
    Rz_robot = np.array([
        [np.cos(yaw_robot), -np.sin(yaw_robot), 0],
        [np.sin(yaw_robot),  np.cos(yaw_robot), 0],
        [0,                  0,                 1],
    ]) @ np.diag([1, -1, -1])

    t_robot = np.eye(4)
    t_robot[:3, :3] = Rz_robot
    t_robot[:3,  3] = [center_xy[0], center_xy[1], center_z]

    t_cam = T_robot_cam @ t_robot

    footprint_area_m2 = float(w * h)
    square_ratio = float(min(w, h) / max(w, h)) if max(w, h) > 1e-6 else 0.0

    return {
        'colour': colour,
        'size_m': estimated_size_m,
        't_robot': t_robot,
        't_cam': t_cam,
        'image_center_px': None,
        'yaw_deg': angle_deg,
        'rect_w_m': float(w),
        'rect_h_m': float(h),
        'footprint_area_m2': footprint_area_m2,
        'square_ratio': square_ratio,
    }

def detect_visible_top_blocks(image, point_cloud, t_cam_robot, camera_intrinsic):
    if image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr = image

    hsv = to_hsv(image)
    full_mask = build_full_mask(hsv, bgr)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(full_mask)

    blocks = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < MIN_COMPONENT_AREA:
            continue

        comp_mask = (labels == i).astype(np.uint8)

        if not is_top_face(comp_mask, point_cloud, t_cam_robot):
            continue

        colour = dominant_colour(hsv, comp_mask)
        result = estimate_block_pose(comp_mask, point_cloud, t_cam_robot, colour)
        if result is None:
            continue

        pt = result['t_cam'][:3, 3]
        if pt[2] <= 0:
            continue

        K = camera_intrinsic
        u = int(K[0, 0] * pt[0] / pt[2] + K[0, 2])
        v = int(K[1, 1] * pt[1] / pt[2] + K[1, 2])
        result['image_center_px'] = (u, v)

        blocks.append(result)

    cv2.imshow('All Masks', full_mask)
    return blocks

# -------------------------------------------------
# Block selection
# -------------------------------------------------

def choose_leftmost_red(blocks):
    reds = [b for b in blocks if b['colour'] == 'red']
    if not reds:
        return None
    return min(reds, key=lambda b: b['image_center_px'][0])

def choose_green(blocks, image_shape):
    greens = [b for b in blocks if b['colour'] == 'green']
    if not greens:
        return None

    fully_visible = [b for b in greens if green_top_face_fully_visible(b, image_shape)]
    if not fully_visible:
        return None

    return min(fully_visible, key=lambda b: b['image_center_px'][0])

# -------------------------------------------------
# Target pose builders
# -------------------------------------------------

def euler_deg_to_pose_xyz_mm(x_mm, y_mm, z_mm, r_deg, p_deg, yaw_deg):
    T = np.eye(4)
    R = Rotation.from_euler('xyz', [r_deg, p_deg, yaw_deg], degrees=True).as_matrix()
    T[:3, :3] = R
    T[:3,  3] = np.array([x_mm, y_mm, z_mm], dtype=float) / 1000.0
    return T

def make_red_drop_pose(red_index):
    """
    red_index = 0 for first red placed,
                1 for second red placed, etc.
    Each next cube shifts +27 mm in X.
    """
    x_mm = RED_START_X_MM + red_index * RED_STEP_X_MM
    y_mm = RED_START_Y_MM
    z_mm = RED_PLACE_Z_MM
    return euler_deg_to_pose_xyz_mm(
        x_mm, y_mm, z_mm,
        RED_R_DEG, RED_P_DEG, RED_YAW_DEG
    )

def make_green_drop_pose():
    return euler_deg_to_pose_xyz_mm(
        GREEN_PLACE_X_MM, GREEN_PLACE_Y_MM, GREEN_PLACE_Z_MM,
        GREEN_R_DEG, GREEN_P_DEG, GREEN_YAW_DEG
    )

# -------------------------------------------------
# Pick/place wrappers
# -------------------------------------------------

def pick_block(arm, block):
    if block['colour'] == 'green':
        grasp_cube_small(arm, block['t_robot'], block['size_m'])
    else:
        grasp_cube_large(arm, block['t_robot'], block['size_m'])

def place_block(arm, target_pose, size_m):
    place_cube(arm, target_pose, size_m)

# -------------------------------------------------
# Preview
# -------------------------------------------------

def preview_blocks(image, camera_intrinsic, blocks, title):
    canvas = image.copy()
    for b in blocks:
        draw_pose_axes(canvas, camera_intrinsic, b['t_cam'])
        u, v = b['image_center_px']
        label = f"{b['colour']} x={u} yaw={b['yaw_deg']:.1f}"
        cv2.putText(canvas, label, (u, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 2, cv2.LINE_AA)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1280, 720)
    cv2.imshow(title, canvas)

# -------------------------------------------------
# Main loop
# -------------------------------------------------
def clear_reds_until_green(arm, zed):
    red_place_count = 0

    while True:
        print(f"\n=== Cycle {red_place_count + 1} ===")

        cv_image, point_cloud = zed.get_synchronized_frame()
        t_cam_robot = get_transform_camera_robot(cv_image, zed.camera_intrinsic)
        if t_cam_robot is None:
            print("Arena transform not found.")
            return False
        print("About to voxelize...")
        voxel_grid = voxelize_between_three_tags(
            image=cv_image,
            point_cloud=point_cloud,
            camera_intrinsic=zed.camera_intrinsic,
            T_cam_robot=t_cam_robot,
            region_tag_ids=[0, 1, 2],   # change to your actual 3 tag IDs
            tag_size=0.055,             # change if your tags are a different size
            voxel_size=0.02,            # 20 mm
            z_min=-0.01,
            z_max=0.25,
            show=True
        )
        print("Voxelization finished.")

        blocks = detect_visible_top_blocks(cv_image, point_cloud, t_cam_robot, zed.camera_intrinsic)
        if not blocks:
            print("No visible top blocks found.")
            return False

        if PREVIEW:
            preview_blocks(cv_image, zed.camera_intrinsic, blocks, 'Visible blocks')
        print("Click the image window, then press 'k' to continue.")
        key = cv2.waitKey(0) & 0xFF
        print(f"Key pressed: {key}")
        if key != ord('k'):
            print("Aborted by user.")
            return False

        green_block = choose_green(blocks, cv_image.shape)
        if green_block is not None:
            print("Green block visible. Picking green block.")
            print(f"Detected green yaw = {green_block['yaw_deg']:.2f} deg")

            pick_block(arm, green_block)

            green_target = make_green_drop_pose()
            place_block(arm, green_target, green_block['size_m'])

            arm.set_position(z=20, relative=True, wait=True)
            return True

        red_block = choose_leftmost_red(blocks)
        if red_block is None:
            print("No red block visible and green not visible.")
            return False

        target_red_pose = make_red_drop_pose(red_place_count)

        print(
            f"Moving red block #{red_place_count + 1} "
            f"to ({RED_START_X_MM + red_place_count * RED_STEP_X_MM}, "
            f"{RED_START_Y_MM}, {RED_PLACE_Z_MM}) mm"
        )

        pick_block(arm, red_block)
        place_block(arm, target_red_pose, red_block['size_m'])

        red_place_count += 1

        arm.set_position(z=20, relative=True, wait=True)
        time.sleep(0.2)

        

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
        success = clear_reds_until_green(arm, zed)
        print(f"\nFinished. success={success}")
        arm.move_gohome(wait=True, speed=500, mvacc=700)

    finally:
        cv2.destroyAllWindows()
        arm.disconnect()
        zed.close()

if __name__ == "__main__":
    main()