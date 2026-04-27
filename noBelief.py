"""
simple_left_to_right.py

Simplified shell game script:
- Voxelize to detect cup locations
- Plan search path using nearest-neighbor (start leftmost, always go to closest)
- Check each cup sequentially until red cube is found
- No belief logic, no audio sweep
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy
import numpy as np
import open3d as o3d
from xarm.wrapper import XArmAPI
from pupil_apriltags import Detector

# Import project modules
from sweep import ROBOT_IP, GRIPPER_LENGTH
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from voxel import voxelize_table
from grasp_and_rotate import grasp_and_rotate as do_grasp_rotate
from push_cube import push_cube as do_push_cube


# ── Configuration ────────────────────────────────────────────────────────────

CUBE_TAG_FAMILY = "tag36h11"
CUBE_TAG_SIZE   = 0.0208  # meters
RED_SETTLE_SECONDS = 0.6
RED_PIXEL_THRESHOLD = 1500
RED_KERNEL_SIZE = 5

# Initialize AprilTag detector
cube_detector = Detector(families=CUBE_TAG_FAMILY)


# ── Visualization ────────────────────────────────────────────────────────────

def visualize_cups(voxel_grid, cup_positions, title="Detected Cups"):
    """
    Visualize the voxel grid with cup position markers and search path.
    
    Args:
        voxel_grid: Open3D VoxelGrid
        cup_positions: List of (x, y, z) positions in meters for each cup (in search order)
        title: Window title
    """
    # Create point cloud from voxel grid
    voxels = voxel_grid.get_voxels()
    voxel_size = voxel_grid.voxel_size
    origin = np.asarray(voxel_grid.origin)
    
    positions = []
    colors = []
    for voxel in voxels:
        grid_index = np.array(voxel.grid_index, dtype=np.int32)
        pos = origin + (grid_index + 0.5) * voxel_size
        positions.append(pos)
        colors.append(voxel.color)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(positions))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    
    # Create voxel grid visualization
    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    
    # Create coordinate frame
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # Create sphere markers for cup positions and path lines
    geometries = [vg, axes]
    
    # Add spheres for each cup
    for i, pos in enumerate(cup_positions):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        sphere.translate(pos)
        # Color code: search order = blue -> red gradient
        t = i / max(1, len(cup_positions) - 1)
        color = [t, 0.2, 1.0 - t]  # Blue -> Purple -> Red
        sphere.paint_uniform_color(color)
        geometries.append(sphere)
    
    # Add lines connecting cups in search order
    if len(cup_positions) > 1:
        line_points = np.array(cup_positions)
        lines = [[i, i+1] for i in range(len(cup_positions) - 1)]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[0.9, 0.9, 0.0] for _ in lines])  # Yellow lines
        geometries.append(line_set)
    
    print(f"[Visualize] {title}")
    print(f"  Search order: {len(cup_positions)} cups")
    total_distance = 0.0
    for i, pos in enumerate(cup_positions):
        if i > 0:
            dist = np.linalg.norm(pos - cup_positions[i-1]) * 1000
            total_distance += dist
            print(f"    {i+1}. ({pos[0]*1000:.1f}, {pos[1]*1000:.1f}, {pos[2]*1000:.1f}) mm  [+{dist:.1f} mm from prev]")
        else:
            print(f"    {i+1}. ({pos[0]*1000:.1f}, {pos[1]*1000:.1f}, {pos[2]*1000:.1f}) mm  [START]")
    if len(cup_positions) > 1:
        print(f"  Total path length: {total_distance:.1f} mm")
    
    o3d.visualization.draw_geometries(geometries, window_name=title, width=1280, height=720)


# ── Helper functions ─────────────────────────────────────────────────────────

def capture_frame():
    """Capture a single frame from ZED camera."""
    zed = ZedCamera()
    try:
        cv_image, point_cloud = zed.get_synchronized_frame()
        K = zed.camera_intrinsic
        return cv_image, point_cloud, K
    finally:
        zed.close()


def detect_all_tags(cv_image, camera_intrinsic, T_cam_robot):
    """Detect all AprilTags and return their transforms in robot frame."""
    # Convert to grayscale
    if cv_image.shape[2] == 4:
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    gray = numpy.ascontiguousarray(gray, dtype=numpy.uint8)

    # Extract camera parameters
    fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
    cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]

    # Detect tags
    tags = cube_detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(fx, fy, cx, cy),
        tag_size=CUBE_TAG_SIZE,
    )

    # Transform tags to robot frame
    T_cam_robot_inv = numpy.linalg.inv(T_cam_robot)
    results = []
    for tag in tags:
        T_cam_cube = numpy.eye(4)
        T_cam_cube[:3, :3] = tag.pose_R
        T_cam_cube[:3, 3] = tag.pose_t.flatten()
        T_robot_cube = T_cam_robot_inv @ T_cam_cube
        results.append((tag.tag_id, T_robot_cube))
    
    return results


def detect_red_cube_fullframe(cv_image, red_threshold=RED_PIXEL_THRESHOLD):
    """Detect red cube using full-frame HSV color detection."""
    if cv_image.shape[2] == 4:
        bgr = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
    else:
        bgr = cv_image

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    # Red color ranges (wraps around at 0/180)
    m1 = cv2.inRange(hsv, np.array([0,   120, 50]),  np.array([10,  255, 255]))
    m2 = cv2.inRange(hsv, np.array([160, 120, 50]),  np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(m1, m2)

    # Morphological opening to reduce noise
    kernel = np.ones((RED_KERNEL_SIZE, RED_KERNEL_SIZE), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    red_pixels = int(cv2.countNonZero(red_mask))
    print(red_pixels)
    found = red_pixels > red_threshold
    return found, red_pixels


def safe_gohome(arm):
    """Safely return arm to home position."""
    code, pos = arm.get_position()
    if code == 0 and pos:
        arm.set_position(pos[0], pos[1], pos[2] + 20, pos[3], pos[4], pos[5], wait=True)
    arm.move_gohome(wait=True)


# ── Main execution ───────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SIMPLE LEFT-TO-RIGHT SWEEP")
    print("=" * 60)
    print()

    # ─── Wall-clock timer: start ─────────────────────────────────────────────
    _t_start = time.perf_counter()
    # ─────────────────────────────────────────────────────────────────────────

    # Initialize robot
    print("Setting up robot...")
    arm = XArmAPI(ROBOT_IP)
    arm.connect()
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)
    print("✓ Robot ready\n")

    try:
        # Step 1: Capture initial frame and detect tags
        print("─" * 60)
        print("STEP 1 — Detecting cups")
        print("─" * 60)
        
        cv_image, point_cloud, K = capture_frame()
        print("✓ Frame captured")

        # Get camera-robot transform
        T_cam_robot = get_transform_camera_robot(cv_image, K)
        if T_cam_robot is None:
            raise RuntimeError("Could not compute camera-robot transform.")
        print("✓ Camera-robot transform computed")

        # Detect all tags (cups)
        tag_results = detect_all_tags(cv_image, K, T_cam_robot)
        if not tag_results:
            raise RuntimeError("No AprilTags detected! Make sure cups are visible.")
        
        # Filter out tags that are too high (false detections on walls, shelves, etc.)
        Z_MIN = 0.05   # 50mm - table surface
        Z_MAX = 0.20   # 200mm - maximum cup height
        
        filtered_results = []
        for tag_id, T_robot_cube in tag_results:
            z_height = T_robot_cube[2, 3]
            if Z_MIN <= z_height <= Z_MAX:
                filtered_results.append((tag_id, T_robot_cube))
            else:
                print(f"  ⚠ Filtered out tag {tag_id} at height {z_height*1000:.1f} mm (outside {Z_MIN*1000:.0f}-{Z_MAX*1000:.0f} mm range)")
        
        tag_results = filtered_results
        if not tag_results:
            raise RuntimeError("No valid cups detected after filtering!")
        
        print(f"✓ Detected {len(tag_results)} valid cup(s)\n")

        # Step 2: Sort cups using nearest-neighbor (for scattered arrangement)
        print("─" * 60)
        print("STEP 2 — Planning search path (nearest-neighbor)")
        print("─" * 60)
        
        # Nearest-neighbor traversal: start leftmost, always go to closest unchecked
        n_cups = len(tag_results)
        visited = [False] * n_cups
        sorted_tags = []
        
        # Start with leftmost cup (min X)
        current_idx = min(range(n_cups), key=lambda i: tag_results[i][1][0, 3])
        visited[current_idx] = True
        sorted_tags.append(tag_results[current_idx])
        
        tag_id, T = tag_results[current_idx]
        print(f"  Start: tag {tag_id} at ({T[0, 3]*1000:.1f}, {T[1, 3]*1000:.1f}) mm")
        
        # Keep picking nearest unvisited cup
        for _ in range(n_cups - 1):
            current_pos = tag_results[current_idx][1][:2, 3]
            
            # Find nearest unvisited cup
            best_idx = None
            best_dist = float('inf')
            for i in range(n_cups):
                if not visited[i]:
                    dist = np.linalg.norm(tag_results[i][1][:2, 3] - current_pos)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
            
            if best_idx is None:
                break
            
            visited[best_idx] = True
            sorted_tags.append(tag_results[best_idx])
            
            tag_id, T = tag_results[best_idx]
            print(f"  → tag {tag_id} at ({T[0, 3]*1000:.1f}, {T[1, 3]*1000:.1f}) mm  (distance: {best_dist*1000:.1f} mm)")
            
            current_idx = best_idx
        
        print()
        
        # Step 2b: Visualize the detected cups and search path
        print("─" * 60)
        print("STEP 2b — Visualizing search path")
        print("─" * 60)
        
        # Create voxel grid for visualization
        voxel_grid = voxelize_table(point_cloud, T_cam_robot)
        if voxel_grid is None:
            print("⚠ Warning: Could not create voxel grid for visualization")
        else:
            # Extract cup positions in search order
            cup_positions = [T_robot_cube[:3, 3] for (_, T_robot_cube) in sorted_tags]
            visualize_cups(voxel_grid, cup_positions, title="Search Path: Nearest-Neighbor Order")
        print()

        # Step 3: Check each cup left to right
        found_target = False
        target_transform = None
        
        for i, (tag_id, T_robot_cube) in enumerate(sorted_tags):
            print("─" * 60)
            print(f"STEP {i+3} — Checking cup {i+1}/{len(sorted_tags)} (tag {tag_id})")
            print("─" * 60)
            
            x_mm, y_mm = T_robot_cube[0, 3] * 1000, T_robot_cube[1, 3] * 1000
            print(f"Position: ({x_mm:.1f}, {y_mm:.1f}) mm")
            
            # Grasp and rotate the cup
            print("Grasping and rotating...")
            do_grasp_rotate(arm, T_robot_cube, rotate_deg=180.0)
            safe_gohome(arm)
            
            # Wait for settling
            time.sleep(RED_SETTLE_SECONDS)
            
            # Capture new frame and check for red cube
            print("Checking for red cube...")
            observe_cv, _, _ = capture_frame()
            found, red_pixels = detect_red_cube_fullframe(observe_cv)
            
            print(f"Red pixels: {red_pixels} (threshold: {RED_PIXEL_THRESHOLD})")
            
            if found:
                print("✓ RED CUBE FOUND!")
                found_target = True
                target_transform = T_robot_cube
                break
            else:
                print("✗ No red cube under this cup")
            print()

        # Step 4: Push the target cube if found
        if found_target and target_transform is not None:
            print("─" * 60)
            print("FINAL STEP — Pushing the target cube")
            print("─" * 60)
            
            do_push_cube(arm, target_transform, target_xy_mm=None)
            safe_gohome(arm)
            
            print()
            print("=" * 60)
            print("✓ SUCCESS — Target cube found and pushed!")
            print("=" * 60)
        else:
            print()
            print("=" * 60)
            print("✗ FAILED — Red cube not found under any cup")
            print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\nDisconnecting robot...")
        arm.disconnect()
        # ─── Wall-clock timer: end ───────────────────────────────────────────
        _elapsed = time.perf_counter() - _t_start
        _mins, _secs = divmod(_elapsed, 60)
        print()
        print("=" * 60)
        print(f"Total runtime: {_elapsed:.2f} s  ({int(_mins)} min {_secs:.2f} s)")
        print("=" * 60)
        # ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    main()