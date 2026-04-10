"""
belief.py  —  Discrete Bayes Filter for cup-target localisation, python belief.py --port /dev/ttyACM0

Math (from slide)
    Bel(v_t) = η · P(z_aud | v_t) · P(z_vis | v_t) · Σ_{v_{t-1}} P(v_t | u_t, v_{t-1}) · Bel(v_{t-1})
                                                       ──────────── prediction step ─────────────────
                ──────────────────────── measurement upate ────────────────────────────────────────────

States   v_t   : which cup index holds the target  (discrete, N cups)
Actions  u_t   : NONE | CHECK(i) | PUSH(i)         (discrete)
Obs      z_aud : audio sweep  -> Gaussian spatial likelihood over spikes
         z_vis : camera check -> near-1 (found) or near-0 (not found)

Usage
    bf = BeliefFilter(voxel_grid, n_cups=5)

    # After audio sweep:
    bf.update_audio(xs, ys, mags)

    # Visualise current belief on voxel grid:
    bf.visualise()

    # Get best cup to check:
    cup, prob = bf.best_cup()

    # After robot rotates cup i and camera says NOT found:
    bf.apply_action(Action.CHECK, cup_id=cup.cup_id)
    bf.update_visual(found=False)
    bf.visualise()

    # Repeat until found.  When found:
    bf.apply_action(Action.CHECK, cup_id=cup.cup_id)
    bf.update_visual(found=True)
    # belief collapses to 1.0 on that cup -> push it
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

# Tuneable parameters

# Cup extraction
CUP_Z_THRESHOLD = 0.03  # m  — voxels above this are cup, not table
CUP_CLUSTER_RADIUS = 0.08   # m  — XY radius for flood-fill clustering
MIN_VOXELS_PER_CUP = 5      # discard tiny noise clusters

# Audio likelihood
SPIKE_PERCENTILE = 85      # top N-th percentile of magnitudes = "spike"
SIGMA_SPATIAL = 0.06    # m  — Gaussian falloff from spike to cup centre

# Visual likelihood
P_VIS_FOUND = 0.90 # P(z_vis=found   | target IS   here)
P_VIS_NOT_FOUND = 0.05 # P(z_vis=found   | target NOT  here)  — false positive rate

# Visualisation
TABLE_COLOUR = (0.30, 0.30, 0.32)

# Action enum

class Action(Enum):
    NONE  = auto()   # no robot action — belief prediction is identity
    CHECK = auto()   # cup rotated 180 deg, camera looked inside
    PUSH  = auto()   # cup pushed over, target confirmed

# Cup cluster
@dataclass
class CupCluster:
    cup_id:          int
    voxel_positions: np.ndarray  # (M, 3) metres
    voxel_indices:   List[Tuple] = field(default_factory=list)

    @property
    def centre_xy(self) -> np.ndarray:
        return self.voxel_positions[:, :2].mean(axis=0)

    @property
    def centre_xyz(self) -> np.ndarray:
        return self.voxel_positions.mean(axis=0)

    def __repr__(self):
        cx, cy = self.centre_xy * 1000
        return (f"Cup(id={self.cup_id}, "
                f"xy=({cx:.1f}, {cy:.1f}) mm, "
                f"voxels={len(self.voxel_indices)})")

# Internal: voxel extraction

def _extract_cup_clusters(
    voxel_grid:     o3d.geometry.VoxelGrid,
    z_threshold:    float = CUP_Z_THRESHOLD,
    cluster_radius: float = CUP_CLUSTER_RADIUS,
    min_voxels:     int   = MIN_VOXELS_PER_CUP,
    n_cups:         Optional[int] = None,
) -> List[CupCluster]:
    """Segment cup voxels from the table by height, then cluster in XY."""
    voxels     = voxel_grid.get_voxels()
    voxel_size = voxel_grid.voxel_size
    origin     = np.asarray(voxel_grid.origin)

    indices   = np.array([v.grid_index for v in voxels], dtype=np.int32)
    positions = origin + (indices + 0.5) * voxel_size

    above         = positions[:, 2] > z_threshold
    cup_positions = positions[above]
    cup_indices   = indices[above]

    if len(cup_positions) == 0:
        raise ValueError(
            f"No voxels above z={z_threshold:.3f} m. "
            "Lower CUP_Z_THRESHOLD or check robot-frame Z calibration."
        )

    # Greedy BFS flood-fill in XY
    xy         = cup_positions[:, :2]
    n          = len(xy)
    cluster_id = np.full(n, -1, dtype=np.int32)
    next_id    = 0

    for i in range(n):
        if cluster_id[i] != -1:
            continue
        queue = [i]
        cluster_id[i] = next_id
        head = 0
        while head < len(queue):
            cur  = queue[head]; head += 1
            dists = np.linalg.norm(xy - xy[cur], axis=1)
            nbrs  = np.where((dists < cluster_radius) & (cluster_id == -1))[0]
            cluster_id[nbrs] = next_id
            queue.extend(nbrs.tolist())
        next_id += 1

    clusters: List[CupCluster] = []
    for cid in range(next_id):
        mask = cluster_id == cid
        if mask.sum() < min_voxels:
            continue
        idx_tuples = [tuple(cup_indices[j]) for j in np.where(mask)[0]]
        clusters.append(CupCluster(
            cup_id          = len(clusters),
            voxel_positions = cup_positions[mask],
            voxel_indices   = idx_tuples,
        ))

    # Sort by size, re-number
    clusters.sort(key=lambda c: -len(c.voxel_indices))
    for k, c in enumerate(clusters):
        c.cup_id = k

    if n_cups is not None:
        clusters = clusters[:n_cups]

    print(f"[BeliefFilter] Extracted {len(clusters)} cup(s):")
    for c in clusters:
        print(f"  {c}")

    return clusters

# Internal: colour helpers

def _belief_to_rgb(p: float) -> Tuple[float, float, float]:
    """
    p=0.0 → cold blue  (0.18, 0.27, 0.80)
    p=0.5 → neutral    (0.70, 0.70, 0.70)
    p=1.0 → hot red    (0.85, 0.10, 0.10)
    """
    p       = float(np.clip(p, 0.0, 1.0))
    cold    = np.array([0.18, 0.27, 0.80])
    neutral = np.array([0.70, 0.70, 0.70])
    hot     = np.array([0.85, 0.10, 0.10])
    if p < 0.5:
        t   = p / 0.5
        rgb = (1 - t) * cold + t * neutral
    else:
        t   = (p - 0.5) / 0.5
        rgb = (1 - t) * neutral + t * hot
    return tuple(rgb.tolist())


def _recolour_voxels(
    voxel_grid: o3d.geometry.VoxelGrid,
    cups:       List[CupCluster],
    belief:     np.ndarray,
) -> o3d.geometry.VoxelGrid:
    """Rebuild VoxelGrid with cup voxels painted by belief score."""
    cup_colour_map = {
        tuple(idx): _belief_to_rgb(belief[i])
        for i, cup in enumerate(cups)
        for idx in cup.voxel_indices
    }

    voxel_size = voxel_grid.voxel_size
    origin     = np.asarray(voxel_grid.origin)
    positions, colours = [], []

    for v in voxel_grid.get_voxels():
        idx    = tuple(v.grid_index)
        centre = origin + (np.array(idx) + 0.5) * voxel_size
        colour = cup_colour_map.get(idx, TABLE_COLOUR)
        positions.append(centre)
        colours.append(colour)

    pcd        = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(positions))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colours))
    return o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

# BeliefFilter — stateful Discrete Bayes Filter

class BeliefFilter:
    """
    Discrete Bayes Filter over N cup hypotheses.

    Full update equation:
        Bel(v_t) = η · P(z_aud | v_t) · P(z_vis | v_t)
                     · Σ_{v_{t-1}} P(v_t | u_t, v_{t-1}) · Bel(v_{t-1})

    Methods map to the equation as:
        apply_action(u_t)    →  prediction step  Bel_bar(v_t)
        update_audio(z_aud)  →  audio  likelihood × Bel_bar
        update_visual(z_vis) →  visual likelihood × current Bel, then η
    """

    def __init__(
        self,
        voxel_grid:     o3d.geometry.VoxelGrid,
        n_cups:         int   = 5,
        z_threshold:    float = CUP_Z_THRESHOLD,
        cluster_radius: float = CUP_CLUSTER_RADIUS,
    ):
        self.voxel_grid = voxel_grid
        self.cups       = _extract_cup_clusters(
            voxel_grid,
            z_threshold    = z_threshold,
            cluster_radius = cluster_radius,
            n_cups         = n_cups,
        )
        N              = len(self.cups)
        self.belief    = np.ones(N) / N          # uniform prior Bel(v_0)
        self._ruled_out: set         = set()     # cup_ids confirmed empty
        self._last_action_cup: Optional[int] = None

        print(f"[BeliefFilter] Uniform prior: {1/N*100:.1f}% per cup.\n")


    
    # 1. Prediction step   Bel_bar(v_t) = Σ P(v_t|u_t,v_{t-1}) Bel(v_{t-1})

    def apply_action(self, action: Action, cup_id: Optional[int] = None) -> None:
        """
        Prediction step: propagate belief through the transition model.

        Transition models
        ─────────────────
        Action.NONE
            Identity transition: P(v_t=i | NONE, v_{t-1}=i) = 1
            Bel_bar = Bel  (no change)

        Action.CHECK  (cup_id = i)
            Robot has rotated cup i.  We do not yet know the camera result.
            Transition is still identity here — the visual likelihood step
            (update_visual) handles the information gain.
            Records cup_id so update_visual() knows which cup was checked.

        Action.PUSH  (cup_id = i)
            Target confirmed; belief collapses to delta on cup i.
            Transition: P(v_t=i | PUSH_i, *) = 1, all others 0.
        """
        if action == Action.NONE or cup_id is None:
            return

        if action == Action.PUSH:
            self.belief[:] = 0.0
            idx = self._cup_index(cup_id)
            if idx is not None:
                self.belief[idx] = 1.0
            print(f"[BeliefFilter] PUSH cup {cup_id} → belief collapsed to 1.0.")
            return

        if action == Action.CHECK:
            self._last_action_cup = cup_id
            print(f"[BeliefFilter] CHECK cup {cup_id} registered — "
                  "call update_visual(found=True/False) after camera observation.")


    
    # 2a. Audio measurement update   P(z_aud | v_t)

    def update_audio(
        self,
        xs:   np.ndarray,
        ys:   np.ndarray,
        mags: np.ndarray,
        spike_percentile: float = SPIKE_PERCENTILE,
        sigma_spatial:    float = SIGMA_SPATIAL,
    ) -> None:
        """
        Multiply current belief by the audio likelihood and renormalise.

        Likelihood per cup i:
            L_i = Σ_spikes  mag_s · exp( -‖cup_i.xy − (xs, ys)‖² / 2σ² )

        Only samples in the top `spike_percentile` of magnitudes contribute,
        which acts as a noise gate — low-level ambient signal is ignored.
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
        likelihood = np.zeros(len(self.cups))

        for sx, sy, sm in zip(s_xs, s_ys, s_mags):
            for i, cup in enumerate(self.cups):
                dx = cup.centre_xy[0] - sx
                dy = cup.centre_xy[1] - sy
                likelihood[i] += sm * np.exp(-(dx*dx + dy*dy) / two_sig2)

        # Bayesian update: posterior ∝ likelihood × prior
        self.belief = self.belief * likelihood
        self._normalise()
        self._print_belief("after audio update")


    
    # 2b. Visual measurement update   P(z_vis | v_t)

    def update_visual(self, found: bool) -> None:
        """
        Update belief after camera observes the inside of the last checked cup.

        Sensor model
        ────────────
        For the cup that was checked (cup_id = i):
            P(z=found  | target IS   here) = P_VIS_FOUND      (default 0.95)
            P(z=found  | target NOT  here) = P_VIS_NOT_FOUND  (default 0.05)

        For all other cups:
            P(z=found  | target here) = P_VIS_NOT_FOUND  (false positive)
            P(z=missing| target here) = 1 - P_VIS_NOT_FOUND

        After update:
            found=False → cup added to ruled-out set, belief zeroed there
            found=True  → belief collapses to delta on that cup
        """
        if self._last_action_cup is None:
            raise RuntimeError(
                "update_visual() called with no prior apply_action(CHECK). "
                "Call bf.apply_action(Action.CHECK, cup_id=i) first."
            )

        cup_id = self._last_action_cup
        idx    = self._cup_index(cup_id)
        N      = len(self.cups)

        # Build visual likelihood vector
        # Default: z_vis observation is uninformative for cups not checked
        if found:
            vis_L = np.full(N, P_VIS_NOT_FOUND)          # others: false positive rate
            if idx is not None:
                vis_L[idx] = P_VIS_FOUND                  # checked cup: hit rate
        else:
            vis_L = np.full(N, 1.0 - P_VIS_NOT_FOUND)    # others: true negative
            if idx is not None:
                vis_L[idx] = 1.0 - P_VIS_FOUND            # checked cup: miss rate

        self.belief = self.belief * vis_L

        if not found:
            # Hard rule-out: zero this cup regardless of numerical residual
            if idx is not None:
                self.belief[idx] = 0.0
            self._ruled_out.add(cup_id)
            self._normalise()
            remaining = N - len(self._ruled_out)
            print(f"[BeliefFilter] Visual: cup {cup_id} EMPTY — ruled out. "
                  f"{remaining} cup(s) remaining.")
        else:
            # Collapse to certainty
            self.belief[:] = 0.0
            if idx is not None:
                self.belief[idx] = 1.0
            print(f"[BeliefFilter] Visual: TARGET FOUND under cup {cup_id}!")

        self._last_action_cup = None
        self._print_belief("after visual update")


    
    # Queries

    def best_cup(self) -> Tuple[CupCluster, float]:
        """Return (CupCluster, probability) of the highest-belief active cup."""
        active = [i for i, c in enumerate(self.cups)
                  if c.cup_id not in self._ruled_out]
        if not active:
            raise RuntimeError("All cups ruled out — target not found.")
        best_i = max(active, key=lambda i: self.belief[i])
        return self.cups[best_i], float(self.belief[best_i])

    def ranked_cups(self) -> List[Tuple[CupCluster, float]]:
        """All active cups sorted highest belief first."""
        pairs = [(self.cups[i], float(self.belief[i]))
                 for i in range(len(self.cups))
                 if self.cups[i].cup_id not in self._ruled_out]
        pairs.sort(key=lambda x: -x[1])
        return pairs


    
    # Visualisation
    

    def visualise(
        self,
        window_title: str = "Cup Belief  (blue=unlikely | red=likely)",
    ) -> None:
        """Recolour voxel grid by current belief and open Open3D viewer."""
        coloured_vg = _recolour_voxels(self.voxel_grid, self.cups, self.belief)
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        best, p = self.best_cup()
        print(f"[BeliefFilter] Visualising. Best: {best}  p={p*100:.1f}%")
        print("  Close the Open3D window to continue.\n")
        o3d.visualization.draw_geometries(
            [coloured_vg, axes],
            window_name=window_title,
            width=1280, height=720,
        )


    
    # Internal helpers
    

    def _cup_index(self, cup_id: int) -> Optional[int]:
        for i, c in enumerate(self.cups):
            if c.cup_id == cup_id:
                return i
        return None

    def _normalise(self) -> None:
        total = self.belief.sum()
        if total < 1e-12:
            # Fallback: flat over remaining active cups
            active = [i for i, c in enumerate(self.cups)
                      if c.cup_id not in self._ruled_out]
            self.belief[:] = 0.0
            if active:
                self.belief[np.array(active)] = 1.0 / len(active)
        else:
            self.belief /= total

    def _print_belief(self, tag: str = "") -> None:
        print(f"[BeliefFilter] Belief {tag}:")
        for i, cup in enumerate(self.cups):
            ruled = " ← RULED OUT" if cup.cup_id in self._ruled_out else ""
            bar   = "█" * int(self.belief[i] * 30) + "░" * (30 - int(self.belief[i] * 30))
            cx, cy = cup.centre_xy * 1000
            print(f"  Cup {cup.cup_id} [{bar}] {self.belief[i]*100:5.1f}%"
                  f"  ({cx:.1f}, {cy:.1f}) mm{ruled}")
        print()

#voxelisation

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
    from grasp_and_rotate import grasp_and_rotate as do_grasp_rotate, get_all_cube_poses
    from push_cube import push_cube as do_push_cube

    MIC_PORT = "/dev/ttyACM0"
    CUBE_TAG_FAMILY = 'tag36h11'
    CUBE_TAG_SIZE = 0.0206

    def detect_all_tags(cv_image, camera_intrinsic, T_cam_robot):
        """Detect all AprilTags and return list of (tag_id, t_robot_cube, t_cam_cube)."""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2GRAY) if cv_image.shape[2] == 4 else cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
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
            t_cam_cube[:3, 3] = tag.pose_t.flatten()
            t_robot_cube = T_cam_robot_inv @ t_cam_cube
            results.append((tag.tag_id, t_robot_cube, t_cam_cube))
        return results

    def find_tag_nearest_cup(tag_results, cup_centre_xy_m):
        """Find the AprilTag closest to the cup's XY centre (in metres)."""
        best_dist = float('inf')
        best = None
        for tag_id, t_robot_cube, t_cam_cube in tag_results:
            tag_xy = t_robot_cube[:2, 3]
            d = numpy.linalg.norm(tag_xy - cup_centre_xy_m)
            if d < best_dist:
                best_dist = d
                best = (tag_id, t_robot_cube, t_cam_cube)
        return best, best_dist

    # ── Step 0: Build voxel grid from ZED
    print("=" * 50)
    print("STEP 0: Capture voxel grid from ZED")
    print("=" * 50)
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

    bf = BeliefFilter(voxel_grid, n_cups=5)

    print("─" * 50)
    print("STEP 0b: Initial belief (uniform prior)")
    print("─" * 50)
    bf.visualise("Step 0: Uniform prior")
    time.sleep(1.0)

    # ── Step 1: Audio sweep
    print("─" * 50)
    print("STEP 1: Audio sweep")
    print("─" * 50)

    try:
        x_mm, y_mm, data_x, data_y = sweep_table(arm, port=MIC_PORT)
        arm.move_gohome(wait=True)

        xs = np.array([x_mm / 1000.0])
        ys = np.array([y_mm / 1000.0])
        mags = np.array([100.0])

        print(f"Audio source estimate: ({x_mm:.1f}, {y_mm:.1f}) mm")
        bf.update_audio(xs, ys, mags, spike_percentile=0)
        bf.visualise("Step 1: After audio sweep")

        # ── Steps 2+: Check cups
        step = 2
        while True:
            best_cup, best_p = bf.best_cup()
            print("─" * 50)
            print(f"STEP {step}: Check cup {best_cup.cup_id}  (p={best_p*100:.1f}%)")
            print("─" * 50)

            # Detect AprilTags to find the cube on this cup
            zed = ZedCamera()
            try:
                cv_image, point_cloud = zed.get_synchronized_frame()
                T_cam_robot = get_transform_camera_robot(cv_image, zed.camera_intrinsic)
                if T_cam_robot is None:
                    print("Lost camera-robot transform.")
                    break
                tag_results = detect_all_tags(cv_image, zed.camera_intrinsic, T_cam_robot)
            finally:
                zed.close()

            if not tag_results:
                print("No AprilTags detected.")
                break

            # Match nearest tag to this cup
            cup_xy_m = best_cup.centre_xy  # already in metres
            (tag_id, t_robot_cube, _), dist = find_tag_nearest_cup(tag_results, cup_xy_m)
            dist_mm = dist * 1000.0

            if dist_mm > 80:
                print(f"No tag within 80 mm of cup {best_cup.cup_id} (closest: {dist_mm:.1f} mm)")
                break

            print(f"  Matched tag {tag_id} at {dist_mm:.1f} mm from cup centre")

            # Grasp and rotate 180° using AprilTag pose
            bf.apply_action(Action.CHECK, cup_id=best_cup.cup_id)
            do_grasp_rotate(arm, t_robot_cube, rotate_deg=180.0)
            arm.move_gohome(wait=True)
            time.sleep(0.5)

            # Camera check: look for the red target
            zed = ZedCamera()
            try:
                cv_image2, _ = zed.get_synchronized_frame()
            finally:
                zed.close()

            bgr = cv2.cvtColor(cv_image2, cv2.COLOR_BGRA2BGR) if cv_image2.shape[2] == 4 else cv_image2
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            m1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
            m2 = cv2.inRange(hsv, np.array([160, 80, 80]), np.array([180, 255, 255]))
            red_mask = cv2.bitwise_or(m1, m2)
            kernel = np.ones((5, 5), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            red_pixels = cv2.countNonZero(red_mask)

            RED_THRESHOLD = 300
            found = red_pixels > RED_THRESHOLD
            print(f"  Red pixels: {red_pixels} → {'FOUND' if found else 'NOT FOUND'}")

            bf.update_visual(found=found)
            bf.visualise(f"Step {step}: Cup {best_cup.cup_id} — {'FOUND' if found else 'empty'}")

            if found:
                print(f"\n✓ Target under cup {best_cup.cup_id} → pushing it.")

                # Re-detect with HSV to get target cube pose for pushing
                zed = ZedCamera()
                try:
                    cv_image3, point_cloud3 = zed.get_synchronized_frame()
                    T_cam_robot3 = get_transform_camera_robot(cv_image3, zed.camera_intrinsic)
                    target_cubes = get_all_cube_poses(
                        [cv_image3, point_cloud3], zed.camera_intrinsic, T_cam_robot3
                    )
                finally:
                    zed.close()

                if target_cubes:
                    cup_xy_mm = best_cup.centre_xy * 1000.0
                    target_cube = min(
                        target_cubes,
                        key=lambda tc: numpy.linalg.norm(tc[0][:2, 3] * 1000 - cup_xy_mm)
                    )[0]
                    do_push_cube(arm, target_cube, target_xy_mm=None)
                else:
                    print("Could not re-detect target cube for pushing.")
                break

            step += 1

        arm.move_gohome(wait=True)

    finally:
        arm.disconnect()