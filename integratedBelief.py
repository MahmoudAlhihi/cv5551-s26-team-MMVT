"""
belief_integrated.py  —  Per-voxel Bayes filter for shell-game target localisation
with live ZED stream + live Open3D belief visualisation.

Pipeline
    0. ZED camera stays open for the entire run (OpenCV window shows live feed).
    1. Voxelise the table, instantiate BeliefFilter with uniform per-voxel prior.
    2. Open3D Visualizer window stays open for the entire run, updates in place.
    3. Audio sweep → update_audio() → belief heatmap shifts.
    4. Loop: pick best voxel → find nearest unchecked AprilTag → CHECK (grasp,
       rotate 180°, release, go home) → red-mask check on live frame → either
       soft-decay belief in a radius (not found) or collapse + push (found).

Actions
    Action.NONE        no belief change
    Action.CHECK (xy)  record XY being checked; update_visual() handles the result
    Action.PUSH  (xy)  collapse belief to a delta on the voxel nearest xy

Visual update model (for a CHECK at xy with radius R = CHECK_RADIUS):
    found=True   → belief collapses to 1.0 at the voxel nearest xy
    found=False  → belief inside R is multiplied by P_VIS_NOT_FOUND (soft),
                   belief outside is multiplied by (1 - P_VIS_NOT_FOUND),
                   then renormalised. Mass stays in the distribution.

Usage
    python belief_integrated.py --port /dev/ttyACM0
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from enum import Enum, auto
from typing import List, Optional, Tuple

# ── Tuneable parameters ──────────────────────────────────────────────────────

# Voxel filter
VOXEL_Z_THRESHOLD   = 0.03     # m — only voxels above this get belief

# Audio likelihood
SPIKE_PERCENTILE    = 85
SIGMA_SPATIAL       = 0.06     # m

# Visual likelihood (soft update)
CHECK_RADIUS        = 0.05     # m — radius around a checked XY
P_VIS_FOUND         = 0.90     # P(z=found | target IS at checked xy)
P_VIS_NOT_FOUND     = 0.10     # P(z=found | target NOT at checked xy)

# Red-target detection
RED_PIXEL_THRESHOLD = 300

# Tag matching
TAG_MATCH_MAX_DIST  = 0.10     # m — reject match if nearest tag is farther than this
CHECKED_TAG_RADIUS  = 0.04     # m — tags within this of a previously-checked XY are "already checked"


# ── Action enum ──────────────────────────────────────────────────────────────

class Action(Enum):
    NONE  = auto()
    CHECK = auto()
    PUSH  = auto()


# ── Colour helpers ───────────────────────────────────────────────────────────

_HEATMAP_STOPS = [
    (0.00, np.array([0.00, 0.00, 0.50])),   # dark blue
    (0.25, np.array([0.00, 0.70, 0.90])),   # cyan
    (0.50, np.array([0.00, 0.85, 0.00])),   # green
    (0.75, np.array([0.95, 0.90, 0.00])),   # yellow
    (1.00, np.array([0.90, 0.05, 0.05])),   # red
]


def _belief_to_rgb(p: float) -> Tuple[float, float, float]:
    p = float(np.clip(p, 0.0, 1.0))
    for i in range(len(_HEATMAP_STOPS) - 1):
        lo_p, lo_c = _HEATMAP_STOPS[i]
        hi_p, hi_c = _HEATMAP_STOPS[i + 1]
        if p <= hi_p:
            t = (p - lo_p) / (hi_p - lo_p)
            rgb = (1 - t) * lo_c + t * hi_c
            return tuple(rgb.tolist())
    return tuple(_HEATMAP_STOPS[-1][1].tolist())


# ── BeliefFilter ─────────────────────────────────────────────────────────────

class BeliefFilter:
    """
    Per-voxel discrete Bayes filter with radius-based CHECK updates.

    No cup clustering — belief is a distribution over every voxel above the
    table plane. Actions target (x, y) points in the robot frame rather than
    cup indices.
    """

    def __init__(
        self,
        voxel_grid: o3d.geometry.VoxelGrid,
        z_threshold: float = VOXEL_Z_THRESHOLD,
    ):
        self.voxel_grid = voxel_grid

        voxels     = voxel_grid.get_voxels()
        voxel_size = voxel_grid.voxel_size
        origin     = np.asarray(voxel_grid.origin)

        all_indices   = np.array([v.grid_index for v in voxels], dtype=np.int32)
        all_positions = origin + (all_indices + 0.5) * voxel_size

        above = all_positions[:, 2] > z_threshold
        self.positions = all_positions[above]         # (N, 3) robot-frame metres
        self.indices   = all_indices[above]           # (N, 3) grid indices

        N = len(self.positions)
        if N == 0:
            raise ValueError(
                f"No voxels above z={z_threshold:.3f} m. "
                "Lower VOXEL_Z_THRESHOLD or check robot-frame Z calibration."
            )

        self._idx_map = {tuple(self.indices[i]): i for i in range(N)}

        self.belief   = np.ones(N) / N
        self._has_data = False

        # Action state
        self._last_check_xy: Optional[np.ndarray] = None
        self._checked_xys:   List[np.ndarray]     = []

        print(f"[BeliefFilter] {N} active voxels above z={z_threshold:.3f} m, "
              f"uniform prior {1/N*100:.4f}% each.\n")

    # ── Action interface ─────────────────────────────────────────────────

    def apply_action(
        self,
        action: Action,
        xy: Optional[np.ndarray] = None,
    ) -> None:
        """
        Prediction step of the Bayes filter.

        NONE         → identity; no change.
        CHECK (xy)   → record XY; wait for update_visual() to apply the result.
        PUSH  (xy)   → collapse belief to the voxel nearest xy.
        """
        if action == Action.NONE or xy is None:
            return

        xy = np.asarray(xy, dtype=float).reshape(2)

        if action == Action.CHECK:
            self._last_check_xy = xy.copy()
            print(f"[BeliefFilter] CHECK registered at "
                  f"({xy[0]*1000:.1f}, {xy[1]*1000:.1f}) mm — "
                  "call update_visual(found=...) after observation.")
            return

        if action == Action.PUSH:
            j = self._nearest_voxel(xy)
            self.belief[:] = 0.0
            self.belief[j] = 1.0
            print(f"[BeliefFilter] PUSH at ({xy[0]*1000:.1f}, {xy[1]*1000:.1f}) mm "
                  f"→ belief collapsed to voxel {j}.")
            return

    # ── Audio measurement update ─────────────────────────────────────────

    def update_audio(
        self,
        xs:   np.ndarray,
        ys:   np.ndarray,
        mags: np.ndarray,
        spike_percentile: float = SPIKE_PERCENTILE,
        sigma_spatial:    float = SIGMA_SPATIAL,
    ) -> None:
        """Multiply belief by per-voxel audio likelihood and renormalise."""
        xs   = np.asarray(xs,   dtype=float)
        ys   = np.asarray(ys,   dtype=float)
        mags = np.asarray(mags, dtype=float)

        threshold  = np.percentile(mags, spike_percentile)
        spike_mask = mags >= threshold
        s_xs, s_ys, s_mags = xs[spike_mask], ys[spike_mask], mags[spike_mask]

        print(f"[BeliefFilter] Audio: {spike_mask.sum()} spikes "
              f"(≥{threshold:.1f} @ {spike_percentile:.0f}th pct)")

        two_sig2 = 2.0 * sigma_spatial ** 2
        voxel_xy = self.positions[:, :2]

        likelihood = np.zeros(len(self.positions))
        for sx, sy, sm in zip(s_xs, s_ys, s_mags):
            d2 = (voxel_xy[:, 0] - sx) ** 2 + (voxel_xy[:, 1] - sy) ** 2
            likelihood += sm * np.exp(-d2 / two_sig2)

        self.belief *= likelihood
        self._normalise()
        self._has_data = True
        self._print_belief("after audio update")

    # ── Visual measurement update ────────────────────────────────────────

    def update_visual(
        self,
        found: bool,
        radius: float = CHECK_RADIUS,
    ) -> None:
        """
        Soft Bayesian update on the last CHECK location.

        Likelihood model for each voxel j, given the checked XY:
            if found=True:
                L_j = P_VIS_FOUND       if ‖voxel_j.xy − check_xy‖ < radius
                      P_VIS_NOT_FOUND   otherwise
            if found=False:
                L_j = 1 - P_VIS_FOUND       if inside radius
                      1 - P_VIS_NOT_FOUND   otherwise

        On found=True we additionally collapse to the voxel nearest check_xy
        (treat as certainty, matching the original belief.py semantics).
        """
        if self._last_check_xy is None:
            raise RuntimeError(
                "update_visual() called with no prior apply_action(CHECK, xy)."
            )

        check_xy = self._last_check_xy
        self._checked_xys.append(check_xy.copy())

        voxel_xy = self.positions[:, :2]
        d2       = (voxel_xy[:, 0] - check_xy[0]) ** 2 + (voxel_xy[:, 1] - check_xy[1]) ** 2
        inside   = d2 < (radius ** 2)

        if found:
            # Soft update then collapse (match original "found → certainty").
            L = np.where(inside, P_VIS_FOUND, P_VIS_NOT_FOUND)
            self.belief *= L
            self._normalise()

            j = self._nearest_voxel(check_xy)
            self.belief[:] = 0.0
            self.belief[j] = 1.0
            print(f"[BeliefFilter] Visual: TARGET FOUND near "
                  f"({check_xy[0]*1000:.1f}, {check_xy[1]*1000:.1f}) mm — "
                  f"collapsed to voxel {j}.")
        else:
            # Soft decay inside the checked radius, mass redistributes globally.
            L = np.where(inside, 1.0 - P_VIS_FOUND, 1.0 - P_VIS_NOT_FOUND)
            self.belief *= L
            self._normalise()
            print(f"[BeliefFilter] Visual: NOT FOUND near "
                  f"({check_xy[0]*1000:.1f}, {check_xy[1]*1000:.1f}) mm — "
                  f"soft-decayed inside {radius*1000:.0f} mm.")

        self._last_check_xy = None
        self._has_data = True
        self._print_belief("after visual update")

    # ── Queries ──────────────────────────────────────────────────────────

    def best_voxel(self) -> np.ndarray:
        """Argmax voxel XYZ, robot frame, metres."""
        j   = int(np.argmax(self.belief))
        pos = self.positions[j]
        print(f"[BeliefFilter] Best voxel: idx {j}, "
              f"pos=({pos[0]*1000:.1f}, {pos[1]*1000:.1f}, {pos[2]*1000:.1f}) mm, "
              f"p={self.belief[j]*100:.2f}%")
        return pos

    def top_voxels(self, k: int = 5) -> List[Tuple[np.ndarray, float]]:
        order = np.argsort(-self.belief)[:k]
        return [(self.positions[i].copy(), float(self.belief[i])) for i in order]

    def checked_xys(self) -> List[np.ndarray]:
        return list(self._checked_xys)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _nearest_voxel(self, xy: np.ndarray) -> int:
        d2 = ((self.positions[:, :2] - xy) ** 2).sum(axis=1)
        return int(np.argmin(d2))

    def _normalise(self) -> None:
        total = self.belief.sum()
        if total < 1e-12:
            self.belief[:] = 1.0 / len(self.belief)
        else:
            self.belief /= total

    def _print_belief(self, tag: str = "") -> None:
        top = self.top_voxels(5)
        print(f"[BeliefFilter] Belief {tag}  (top 5 voxels):")
        for rank, (pos, p) in enumerate(top):
            bar = "█" * int(p * 50) + "░" * (50 - int(p * 50))
            print(f"  #{rank} [{bar}] {p*100:6.2f}%  "
                  f"({pos[0]*1000:.1f}, {pos[1]*1000:.1f}, {pos[2]*1000:.1f}) mm")
        print()


# ── Live Open3D visualiser ───────────────────────────────────────────────────

class LiveBeliefVisualiser:
    """
    Non-blocking Open3D window that re-colours a point cloud in place to
    reflect the current belief distribution.

    We rebuild a PointCloud (one point per voxel centre) rather than a
    VoxelGrid because point clouds support in-place colour updates much more
    cleanly than VoxelGrid geometry.
    """

    def __init__(
        self,
        bf: BeliefFilter,
        window_title: str = "Live Belief  (blue=unlikely | red=likely)",
        width: int = 1280,
        height: int = 720,
    ):
        self.bf = bf
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_title, width=width, height=height)

        # Static geometry: full voxel grid in neutral colour (only for context)
        self._build_context_cloud()
        self._build_active_cloud()

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(axes)
        self.vis.add_geometry(self.context_pcd)
        self.vis.add_geometry(self.active_pcd)

        # Marker for best voxel (a small sphere, re-positioned each tick)
        self.marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.012)
        self.marker.paint_uniform_color([1.0, 1.0, 1.0])
        self.marker.compute_vertex_normals()
        self.vis.add_geometry(self.marker)
        self._marker_base = np.asarray(self.marker.vertices).copy()
        self._last_marker_pos = np.zeros(3)

        # One quick render so the window shows up immediately
        self.tick()

    def _build_context_cloud(self) -> None:
        """Grey point cloud of every voxel in the scene (including table)."""
        voxels     = self.bf.voxel_grid.get_voxels()
        voxel_size = self.bf.voxel_grid.voxel_size
        origin     = np.asarray(self.bf.voxel_grid.origin)

        active_set = set(map(tuple, self.bf.indices.tolist()))

        pts, cols = [], []
        for v in voxels:
            idx = tuple(v.grid_index)
            if idx in active_set:
                continue  # active voxels rendered separately with heatmap
            centre = origin + (np.array(idx) + 0.5) * voxel_size
            pts.append(centre)
            cols.append(np.asarray(v.color).tolist())

        self.context_pcd = o3d.geometry.PointCloud()
        if pts:
            self.context_pcd.points = o3d.utility.Vector3dVector(np.array(pts))
            self.context_pcd.colors = o3d.utility.Vector3dVector(np.array(cols))

    def _build_active_cloud(self) -> None:
        """Heatmap point cloud of voxels above the table plane."""
        self.active_pcd = o3d.geometry.PointCloud()
        self.active_pcd.points = o3d.utility.Vector3dVector(self.bf.positions.copy())
        self.active_pcd.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([0.0, 0.0, 0.5]), (len(self.bf.positions), 1))
        )

    def _current_colours(self) -> np.ndarray:
        b = self.bf.belief
        if not self.bf._has_data or b.max() <= 0:
            return np.tile(np.array([0.0, 0.0, 0.5]), (len(b), 1))
        normed = b / b.max()
        return np.array([_belief_to_rgb(p) for p in normed])

    def tick(self) -> None:
        """
        Call this frequently to keep the window responsive.

        Updates colours from current belief and the best-voxel marker position.
        """
        # Update heatmap colours in place
        self.active_pcd.colors = o3d.utility.Vector3dVector(self._current_colours())
        self.vis.update_geometry(self.active_pcd)

        # Move marker to current best voxel
        j       = int(np.argmax(self.bf.belief))
        new_pos = self.bf.positions[j]
        delta   = new_pos - self._last_marker_pos
        if np.linalg.norm(delta) > 1e-6:
            self.marker.translate(delta)
            self._last_marker_pos = new_pos.copy()
            self.vis.update_geometry(self.marker)

        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self) -> None:
        try:
            self.vis.destroy_window()
        except Exception:
            pass


# ── Main pipeline ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    import cv2
    import numpy
    from pupil_apriltags import Detector
    from sweep import sweep_table, ROBOT_IP, GRIPPER_LENGTH
    from xarm.wrapper import XArmAPI
    from utils.zed_camera import ZedCamera
    from checkpoint0 import get_transform_camera_robot
    from voxel import voxelize_table
    from grasp_and_rotate import grasp_and_rotate as do_grasp_rotate
    from push_cube import push_cube as do_push_cube

    MIC_PORT        = "/dev/ttyACM0"
    CUBE_TAG_FAMILY = 'tag36h11'
    CUBE_TAG_SIZE   = 0.0206

    # ── Helpers ──────────────────────────────────────────────────────────

    def detect_all_tags(cv_image, camera_intrinsic, T_cam_robot):
        """Detect all AprilTags → list of (tag_id, T_robot_cube, T_cam_cube)."""
        if cv_image.shape[2] == 4:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        detector = Detector(families=CUBE_TAG_FAMILY)
        fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
        cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]

        tags = detector.detect(gray, estimate_tag_pose=True,
                               camera_params=(fx, fy, cx, cy),
                               tag_size=CUBE_TAG_SIZE)

        T_cam_robot_inv = numpy.linalg.inv(T_cam_robot)
        results = []
        for tag in tags:
            T_cam_cube = numpy.eye(4)
            T_cam_cube[:3, :3] = tag.pose_R
            T_cam_cube[:3, 3]  = tag.pose_t.flatten()
            T_robot_cube = T_cam_robot_inv @ T_cam_cube
            results.append((tag.tag_id, T_robot_cube, T_cam_cube))
        return results

    def find_best_tag(tag_results, target_xy_m, excluded_xys, max_dist):
        """
        Return the (tag, distance) closest to target_xy_m that is NOT within
        CHECKED_TAG_RADIUS of any previously-checked XY.
        Returns (None, inf) if no acceptable tag within max_dist.
        """
        best_dist = float('inf')
        best = None
        for tag_id, T_robot_cube, T_cam_cube in tag_results:
            tag_xy = T_robot_cube[:2, 3]
            # Skip if too close to a previously-checked XY
            already = any(
                numpy.linalg.norm(tag_xy - cxy) < CHECKED_TAG_RADIUS
                for cxy in excluded_xys
            )
            if already:
                continue
            d = numpy.linalg.norm(tag_xy - target_xy_m)
            if d < best_dist:
                best_dist = d
                best = (tag_id, T_robot_cube, T_cam_cube)
        if best is None or best_dist > max_dist:
            return None, best_dist
        return best, best_dist

    def red_mask_count(cv_image):
        """Return number of red pixels on the given BGRA/BGR frame."""
        if cv_image.shape[2] == 4:
            bgr = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
        else:
            bgr = cv_image
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m1  = cv2.inRange(hsv, np.array([  0, 80, 80]), np.array([ 10, 255, 255]))
        m2  = cv2.inRange(hsv, np.array([160, 80, 80]), np.array([180, 255, 255]))
        red = cv2.bitwise_or(m1, m2)
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return cv2.countNonZero(red), red

    def show_frame(cv_image, label=""):
        """Show the ZED frame in an OpenCV window; overlay a label if given."""
        if cv_image.shape[2] == 4:
            disp = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
        else:
            disp = cv_image.copy()
        if label:
            cv2.putText(disp, label, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("ZED 2 - Live", disp)

    def pump(zed, bf_viz, label=""):
        """
        One 'tick' of the live loops: grab a ZED frame, show it, update the
        Open3D belief window, poll GUI events. Call this everywhere we'd
        otherwise time.sleep().
        Returns (cv_image, point_cloud) so caller can use the latest frame.
        """
        cv_image, point_cloud = zed.get_synchronized_frame()
        show_frame(cv_image, label)
        bf_viz.tick()
        # Tiny waitKey keeps OpenCV responsive; 'q' lets user bail out.
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            raise KeyboardInterrupt("User pressed 'q'.")
        return cv_image, point_cloud

    def safe_gohome(arm, zed, bf_viz):
        """Lift 20 mm, then go home, pumping the live windows along the way."""
        code, pos = arm.get_position()
        if code == 0 and pos:
            arm.set_position(pos[0], pos[1], pos[2] + 20,
                             pos[3], pos[4], pos[5], wait=True)
        arm.move_gohome(wait=True)
        for _ in range(10):
            pump(zed, bf_viz, "returning home")

    # ── Step 0: set up robot, ZED, voxel grid, belief filter, viz ────────
    print("=" * 60)
    print("STEP 0 — Setup")
    print("=" * 60)

    arm = XArmAPI(ROBOT_IP)
    arm.connect()
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)

    cv2.namedWindow("ZED 2 - Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ZED 2 - Live", 1280, 720)

    zed = ZedCamera()
    bf_viz: Optional[LiveBeliefVisualiser] = None

    try:
        # Initial frame + voxel grid
        cv_image, point_cloud = zed.get_synchronized_frame()
        show_frame(cv_image, "voxelising")
        cv2.waitKey(1)

        T_cam_robot = get_transform_camera_robot(cv_image, zed.camera_intrinsic)
        if T_cam_robot is None:
            raise RuntimeError("Could not compute camera-robot transform.")
        voxel_grid = voxelize_table(point_cloud, T_cam_robot)
        if voxel_grid is None:
            raise RuntimeError("Voxel grid creation failed.")

        bf     = BeliefFilter(voxel_grid)
        bf_viz = LiveBeliefVisualiser(bf)

        # Warm-up: pump both windows for a moment so the user sees them populate.
        for _ in range(30):
            pump(zed, bf_viz, "uniform prior")

        # ── Step 1: Audio sweep ──────────────────────────────────────────
        print("─" * 60)
        print("STEP 1 — Audio sweep")
        print("─" * 60)

        x_mm, y_mm, _, _ = sweep_table(arm, port=MIC_PORT)
        safe_gohome(arm, zed, bf_viz)

        xs   = np.array([x_mm / 1000.0])
        ys   = np.array([y_mm / 1000.0])
        mags = np.array([100.0])
        print(f"Audio source estimate: ({x_mm:.1f}, {y_mm:.1f}) mm")

        bf.update_audio(xs, ys, mags, spike_percentile=0)
        for _ in range(30):
            pump(zed, bf_viz, "after audio sweep")

        # ── Step 2+: CHECK loop ──────────────────────────────────────────
        step = 2
        found_target = False
        target_tag_T_robot_cube = None

        while not found_target:
            print("─" * 60)
            print(f"STEP {step} — Pick next cube to check")
            print("─" * 60)

            best_xyz = bf.best_voxel()
            best_xy  = best_xyz[:2]

            # Grab a fresh frame for tag detection
            cv_image, _ = pump(zed, bf_viz, f"step {step}: detecting tags")
            T_cam_robot = get_transform_camera_robot(cv_image, zed.camera_intrinsic)
            if T_cam_robot is None:
                print("Lost camera-robot transform.")
                break
            tag_results = detect_all_tags(cv_image, zed.camera_intrinsic, T_cam_robot)
            if not tag_results:
                print("No AprilTags detected — stopping.")
                break

            match, dist = find_best_tag(
                tag_results,
                target_xy_m  = best_xy,
                excluded_xys = bf.checked_xys(),
                max_dist     = TAG_MATCH_MAX_DIST,
            )
            if match is None:
                print(f"No un-checked tag within {TAG_MATCH_MAX_DIST*1000:.0f} mm "
                      f"of best voxel (closest: {dist*1000:.1f} mm). Stopping.")
                break

            tag_id, T_robot_cube, _ = match
            tag_xy = T_robot_cube[:2, 3]
            print(f"  Target tag: id={tag_id}  "
                  f"xy=({tag_xy[0]*1000:.1f}, {tag_xy[1]*1000:.1f}) mm  "
                  f"dist={dist*1000:.1f} mm from best voxel")

            # ── CHECK: register, rotate, observe ──────────────────────────
            bf.apply_action(Action.CHECK, xy=tag_xy)
            bf_viz.tick()

            do_grasp_rotate(arm, T_robot_cube, rotate_deg=180.0)
            safe_gohome(arm, zed, bf_viz)

            # Let the scene settle, then check the live frame for red pixels.
            red_pixels = 0
            for _ in range(20):
                cv_image_post, _ = pump(zed, bf_viz, f"step {step}: observing")
                red_pixels, _ = red_mask_count(cv_image_post)

            found = red_pixels > RED_PIXEL_THRESHOLD
            print(f"  Red pixels: {red_pixels} → {'FOUND' if found else 'NOT FOUND'}")

            bf.update_visual(found=found)
            for _ in range(20):
                pump(zed, bf_viz,
                     f"step {step}: {'FOUND' if found else 'empty'} @ tag {tag_id}")

            if found:
                found_target            = True
                target_tag_T_robot_cube = T_robot_cube
                break

            step += 1

        # ── Final step: push the revealed cube ───────────────────────────
        if found_target and target_tag_T_robot_cube is not None:
            print("─" * 60)
            print("FINAL STEP — Push the target cube")
            print("─" * 60)

            # Commit the PUSH to the belief map for the log.
            final_xy = target_tag_T_robot_cube[:2, 3]
            bf.apply_action(Action.PUSH, xy=final_xy)
            for _ in range(10):
                pump(zed, bf_viz, "pushing target")

            do_push_cube(arm, target_tag_T_robot_cube, target_xy_mm=None)
            safe_gohome(arm, zed, bf_viz)
            print("\n✓ Done. Target cube pushed.")
        else:
            print("\n✗ Target not found. Stopping.")

        # Keep windows open briefly so the user can inspect the final state.
        print("Press 'q' in the ZED window to exit.")
        while True:
            pump(zed, bf_viz, "finished")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        try:
            arm.disconnect()
        except Exception:
            pass
        try:
            zed.close()
        except Exception:
            pass
        if bf_viz is not None:
            bf_viz.close()
        cv2.destroyAllWindows()