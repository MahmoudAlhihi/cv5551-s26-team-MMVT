"""
belief.py  —  Voxel-level Bayes filter for audio-based target localisation

Instead of clustering voxels into cups, we maintain a per-voxel probability
heat map.  After the audio sweep updates the map, we simply pick the ArUco
marker (cube) whose robot-frame XY is closest to the most-probable voxel.

Usage
    bf = BeliefFilter(voxel_grid)
    bf.update_audio(xs, ys, mags)
    bf.visualise()
    best_xyz = bf.best_voxel()          # (3,) metres in robot frame
    # then match to nearest detected ArUco tag
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from typing import List, Optional, Tuple

# ── Tuneable parameters ──────────────────────────────────────────────────────

VOXEL_Z_THRESHOLD  = 0.03     # m — only voxels above this get belief (skip table surface)

# Audio likelihood
SPIKE_PERCENTILE   = 85       # top N-th percentile of magnitudes = "spike"
SIGMA_SPATIAL      = 0.06     # m — Gaussian falloff from spike to voxel centre




# ── Colour helpers ───────────────────────────────────────────────────────────

def _belief_to_rgb(p: float) -> Tuple[float, float, float]:
    """Heatmap: p=0 → dark blue … p=1 → red."""
    p = float(np.clip(p, 0.0, 1.0))
    stops = [
        (0.00, np.array([0.00, 0.00, 0.50])),   # dark blue
        (0.25, np.array([0.00, 0.70, 0.90])),   # cyan
        (0.50, np.array([0.00, 0.85, 0.00])),   # green
        (0.75, np.array([0.95, 0.90, 0.00])),   # yellow
        (1.00, np.array([0.90, 0.05, 0.05])),   # red
    ]
    for i in range(len(stops) - 1):
        lo_p, lo_c = stops[i]
        hi_p, hi_c = stops[i + 1]
        if p <= hi_p:
            t = (p - lo_p) / (hi_p - lo_p)
            rgb = (1 - t) * lo_c + t * hi_c
            return tuple(rgb.tolist())
    return tuple(stops[-1][1].tolist())


def _recolour_voxels(
    voxel_grid: o3d.geometry.VoxelGrid,
    active_indices: dict,          # {(ix,iy,iz): int}  grid_index → belief index
    belief: np.ndarray,
    has_data: bool = False,
) -> o3d.geometry.VoxelGrid:
    """Rebuild VoxelGrid with active voxels painted by belief score.

    Table voxels (not in active_indices) keep their original colour.
    Active voxels are painted by belief:
      - If has_data is False (uniform prior, no sensor readings yet),
        all active voxels are cold blue (p=0).
      - Otherwise, belief is normalised to [0,1] by its max for the heatmap.
    """
    if has_data:
        b_max  = belief.max() if belief.max() > 0 else 1.0
        normed = belief / b_max
    else:
        normed = np.zeros_like(belief)       # everything cold blue

    voxel_size = voxel_grid.voxel_size
    origin     = np.asarray(voxel_grid.origin)
    positions, colours = [], []

    for v in voxel_grid.get_voxels():
        idx    = tuple(v.grid_index)
        centre = origin + (np.array(idx) + 0.5) * voxel_size
        if idx in active_indices:
            colour = _belief_to_rgb(normed[active_indices[idx]])
        else:
            colour = tuple(np.asarray(v.color).tolist())   # keep original colour
        positions.append(centre)
        colours.append(colour)

    pcd        = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(positions))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colours))
    return o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)


# ── BeliefFilter — per-voxel discrete Bayes filter ───────────────────────────

class BeliefFilter:
    """
    Maintains a probability distribution over individual voxels (not cups).

    After an audio sweep, call update_audio() then best_voxel() to get the
    robot-frame XYZ of the most likely target location.
    """

    def __init__(
        self,
        voxel_grid: o3d.geometry.VoxelGrid,
        z_threshold: float = VOXEL_Z_THRESHOLD,
    ):
        self.voxel_grid = voxel_grid

        # Extract every voxel above the table surface
        voxels     = voxel_grid.get_voxels()
        voxel_size = voxel_grid.voxel_size
        origin     = np.asarray(voxel_grid.origin)

        all_indices   = np.array([v.grid_index for v in voxels], dtype=np.int32)
        all_positions = origin + (all_indices + 0.5) * voxel_size

        above_mask     = all_positions[:, 2] > z_threshold
        self.positions = all_positions[above_mask]           # (N, 3) robot-frame metres
        self.indices   = all_indices[above_mask]             # (N, 3) grid indices

        N = len(self.positions)
        if N == 0:
            raise ValueError(
                f"No voxels above z={z_threshold:.3f} m. "
                "Lower VOXEL_Z_THRESHOLD or check robot-frame Z calibration."
            )

        # Map grid_index tuple → belief vector index (for fast recolouring)
        self._idx_map = {tuple(self.indices[i]): i for i in range(N)}

        # Uniform prior — no sensor data yet
        self.belief    = np.ones(N) / N
        self._has_data = False

        print(f"[BeliefFilter] {N} active voxels above z={z_threshold:.3f} m, "
              f"uniform prior {1/N*100:.4f}% each.\n")

    # ── Audio measurement update ─────────────────────────────────────────

    def update_audio(
        self,
        xs:   np.ndarray,
        ys:   np.ndarray,
        mags: np.ndarray,
        spike_percentile: float = SPIKE_PERCENTILE,
        sigma_spatial:    float = SIGMA_SPATIAL,
    ) -> None:
        """
        Multiply current belief by per-voxel audio likelihood and renormalise.

        Likelihood per voxel j:
            L_j = Σ_spikes  mag_s · exp( -‖voxel_j.xy − (xs, ys)‖² / 2σ² )
        """
        xs   = np.asarray(xs,   dtype=float)
        ys   = np.asarray(ys,   dtype=float)
        mags = np.asarray(mags, dtype=float)

        threshold  = np.percentile(mags, spike_percentile)
        spike_mask = mags >= threshold
        s_xs       = xs[spike_mask]
        s_ys       = ys[spike_mask]
        s_mags     = mags[spike_mask]

        print(f"[BeliefFilter] Audio: {spike_mask.sum()} spikes "
              f"(≥{threshold:.1f} @ {spike_percentile:.0f}th pct)")

        two_sig2   = 2.0 * sigma_spatial ** 2
        voxel_xy   = self.positions[:, :2]                   # (N, 2)

        # Vectorised: accumulate Gaussian-weighted magnitude from each spike
        likelihood = np.zeros(len(self.positions))
        for sx, sy, sm in zip(s_xs, s_ys, s_mags):
            d2 = (voxel_xy[:, 0] - sx) ** 2 + (voxel_xy[:, 1] - sy) ** 2
            likelihood += sm * np.exp(-d2 / two_sig2)

        # Bayesian update: posterior ∝ likelihood × prior
        self.belief *= likelihood
        self._normalise()
        self._has_data = True
        self._print_belief("after audio update")

    # ── Queries ──────────────────────────────────────────────────────────

    def best_voxel(self) -> np.ndarray:
        """Return the (3,) robot-frame XYZ of the highest-belief voxel."""
        best_i = int(np.argmax(self.belief))
        pos    = self.positions[best_i]
        print(f"[BeliefFilter] Best voxel: index {best_i}, "
              f"pos=({pos[0]*1000:.1f}, {pos[1]*1000:.1f}, {pos[2]*1000:.1f}) mm, "
              f"p={self.belief[best_i]*100:.2f}%")
        return pos

    def top_voxels(self, k: int = 5) -> List[Tuple[np.ndarray, float]]:
        """Return the top-k voxels as [(xyz, probability), ...]."""
        order = np.argsort(-self.belief)[:k]
        return [(self.positions[i].copy(), float(self.belief[i])) for i in order]

    # ── Visualisation ────────────────────────────────────────────────────

    def visualise(
        self,
        window_title: str = "Voxel Belief  (blue=unlikely | red=likely)",
    ) -> None:
        """Recolour voxel grid by current belief and open Open3D viewer."""
        coloured_vg = _recolour_voxels(
            self.voxel_grid, self._idx_map, self.belief, self._has_data
        )
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        best_pos = self.positions[int(np.argmax(self.belief))]
        best_p   = self.belief.max()
        print(f"[BeliefFilter] Visualising. Best voxel at "
              f"({best_pos[0]*1000:.1f}, {best_pos[1]*1000:.1f}, {best_pos[2]*1000:.1f}) mm  "
              f"p={best_p*100:.2f}%")
        print("  Close the Open3D window to continue.\n")
        o3d.visualization.draw_geometries(
            [coloured_vg, axes],
            window_name=window_title,
            width=1280, height=720,
        )

    # ── Internal helpers ─────────────────────────────────────────────────

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
    from push_cube import push_cube as do_push_cube

    MIC_PORT        = "/dev/ttyACM0"
    CUBE_TAG_FAMILY = 'tag36h11'
    CUBE_TAG_SIZE   = 0.0206

    # ── Helpers ──────────────────────────────────────────────────────────

    def detect_all_tags(cv_image, camera_intrinsic, T_cam_robot):
        """Detect all AprilTags and return list of (tag_id, t_robot_cube, t_cam_cube)."""
        gray = (cv2.cvtColor(cv_image, cv2.COLOR_BGRA2GRAY)
                if cv_image.shape[2] == 4
                else cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY))
        detector = Detector(families=CUBE_TAG_FAMILY)
        fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
        cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]

        tags = detector.detect(gray, estimate_tag_pose=True,
                               camera_params=(fx, fy, cx, cy),
                               tag_size=CUBE_TAG_SIZE)

        T_cam_robot_inv = numpy.linalg.inv(T_cam_robot)
        results = []
        for tag in tags:
            t_cam_cube = numpy.eye(4)
            t_cam_cube[:3, :3] = tag.pose_R
            t_cam_cube[:3, 3]  = tag.pose_t.flatten()
            t_robot_cube = T_cam_robot_inv @ t_cam_cube
            results.append((tag.tag_id, t_robot_cube, t_cam_cube))
        return results

    def find_tag_nearest_point(tag_results, target_xy_m):
        """Find the AprilTag closest to a target XY point (in metres)."""
        best_dist = float('inf')
        best = None
        for tag_id, t_robot_cube, t_cam_cube in tag_results:
            tag_xy = t_robot_cube[:2, 3]
            d = numpy.linalg.norm(tag_xy - target_xy_m)
            if d < best_dist:
                best_dist = d
                best = (tag_id, t_robot_cube, t_cam_cube)
        return best, best_dist

    def safe_gohome(arm):
        """Lift 20 mm from current position, then go home."""
        code, pos = arm.get_position()
        if code == 0 and pos:
            arm.set_position(pos[0], pos[1], pos[2] + 20,
                             pos[3], pos[4], pos[5], wait=True)
        arm.move_gohome(wait=True)

    # ── Step 0: Build voxel grid from ZED ────────────────────────────────
    print("=" * 60)
    print("STEP 0 — Capture voxel grid from ZED")
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

    zed = ZedCamera()
    try:
        cv_image, point_cloud = zed.get_synchronized_frame()
        T_cam_robot = get_transform_camera_robot(cv_image, zed.camera_intrinsic)
        if T_cam_robot is None:
            raise RuntimeError("Could not compute camera-robot transform.")
        voxel_grid = voxelize_table(point_cloud, T_cam_robot)
        if voxel_grid is None:
            raise RuntimeError("Voxel grid creation failed.")
    finally:
        zed.close()

    bf = BeliefFilter(voxel_grid)

    print("─" * 60)
    print("STEP 0b — Initial belief (uniform prior)")
    print("─" * 60)
    bf.visualise("Step 0: Uniform prior")
    time.sleep(1.0)

    # ── Step 1: Audio sweep ──────────────────────────────────────────────
    print("─" * 60)
    print("STEP 1 — Audio sweep")
    print("─" * 60)

    try:
        x_mm, y_mm, data_x, data_y = sweep_table(arm, port=MIC_PORT)
        safe_gohome(arm)

        xs   = np.array([x_mm / 1000.0])
        ys   = np.array([y_mm / 1000.0])
        mags = np.array([100.0])

        print(f"Audio source estimate: ({x_mm:.1f}, {y_mm:.1f}) mm")
        bf.update_audio(xs, ys, mags, spike_percentile=0)
        bf.visualise("Step 1: After audio sweep")

        # ── Step 2: Pick the ArUco marker nearest to the best voxel ──────
        print("─" * 60)
        print("STEP 2 — Match best voxel to nearest ArUco marker")
        print("─" * 60)

        best_xyz = bf.best_voxel()            # (3,) metres, robot frame
        best_xy  = best_xyz[:2]               # only need XY for matching

        zed = ZedCamera()
        try:
            cv_image, point_cloud = zed.get_synchronized_frame()
            T_cam_robot = get_transform_camera_robot(cv_image, zed.camera_intrinsic)
            if T_cam_robot is None:
                raise RuntimeError("Lost camera-robot transform.")
            tag_results = detect_all_tags(cv_image, zed.camera_intrinsic, T_cam_robot)
        finally:
            zed.close()

        if not tag_results:
            raise RuntimeError("No AprilTags detected — cannot match to voxel.")

        print(f"  Detected {len(tag_results)} ArUco tag(s)")

        (tag_id, t_robot_cube, t_cam_cube), dist = find_tag_nearest_point(
            tag_results, best_xy
        )
        dist_mm = dist * 1000.0
        tag_xy  = t_robot_cube[:2, 3] * 1000.0

        print(f"  Nearest tag: id={tag_id}  "
              f"xy=({tag_xy[0]:.1f}, {tag_xy[1]:.1f}) mm  "
              f"dist={dist_mm:.1f} mm from best voxel")

        # ── Step 3: Push that cube ───────────────────────────────────────
        print("─" * 60)
        print(f"STEP 3 — Push cube (tag {tag_id})")
        print("─" * 60)

        do_push_cube(arm, t_robot_cube, target_xy_mm=None)
        safe_gohome(arm)
        print("\n✓ Done.")

    finally:
        arm.disconnect()