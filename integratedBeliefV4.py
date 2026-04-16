"""
belief_integrated_simple.py

Simplified stable version:
- Keeps region-based voxel belief logic from the integrated script
- Keeps full-frame HSV red detection from the old stable script
- Removes:
    * background ZED capture thread
    * live cv2 display pump
    * separate Open3D viz process
- Uses the old stable camera access pattern:
    open ZED -> get one frame -> close ZED
"""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d


# ── Tuneable parameters ──────────────────────────────────────────────────────

VOXEL_Z_THRESHOLD   = 0.03
SPIKE_PERCENTILE    = 85
SIGMA_SPATIAL       = 0.06

REGION_CLUSTER_RADIUS = 0.08
REGION_MIN_VOXELS     = 5

P_VIS_FOUND         = 0.90
P_VIS_NOT_FOUND     = 0.10

RED_SETTLE_SECONDS  = 0.6
RED_PIXEL_THRESHOLD = 300
RED_KERNEL_SIZE     = 5

TAG_MATCH_MAX_DIST  = 0.10
CHECKED_TAG_RADIUS  = 0.04


class Action(Enum):
    NONE  = auto()
    CHECK = auto()
    PUSH  = auto()


# ── Colour helpers ───────────────────────────────────────────────────────────

_HEATMAP_STOPS = [
    (0.00, np.array([0.00, 0.00, 0.50])),
    (0.25, np.array([0.00, 0.70, 0.90])),
    (0.50, np.array([0.00, 0.85, 0.00])),
    (0.75, np.array([0.95, 0.90, 0.00])),
    (1.00, np.array([0.90, 0.05, 0.05])),
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


# ── Region ───────────────────────────────────────────────────────────────────

class Region:
    __slots__ = ("region_id", "voxel_idx", "centre_xy", "centre_xyz")

    def __init__(self, region_id: int, voxel_idx: np.ndarray,
                 centre_xy: np.ndarray, centre_xyz: np.ndarray):
        self.region_id  = region_id
        self.voxel_idx  = voxel_idx
        self.centre_xy  = centre_xy
        self.centre_xyz = centre_xyz

    def __repr__(self):
        cx, cy = self.centre_xy * 1000
        return f"Region(id={self.region_id}, xy=({cx:.1f}, {cy:.1f}) mm, voxels={len(self.voxel_idx)})"


def _flood_fill_xy(
    positions: np.ndarray,
    cluster_radius: float,
    min_voxels: int,
) -> List[np.ndarray]:
    xy         = positions[:, :2]
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
            cur = queue[head]
            head += 1
            dists = np.linalg.norm(xy - xy[cur], axis=1)
            nbrs  = np.where((dists < cluster_radius) & (cluster_id == -1))[0]
            cluster_id[nbrs] = next_id
            queue.extend(nbrs.tolist())
        next_id += 1

    raw = []
    for cid in range(next_id):
        member = np.where(cluster_id == cid)[0]
        if len(member) >= min_voxels:
            raw.append(member)
    raw.sort(key=lambda m: -len(m))
    return raw


# ── Belief filter ────────────────────────────────────────────────────────────

class BeliefFilter:
    def __init__(self, voxel_grid: o3d.geometry.VoxelGrid, z_threshold: float = VOXEL_Z_THRESHOLD):
        self.voxel_grid = voxel_grid

        voxels     = voxel_grid.get_voxels()
        voxel_size = voxel_grid.voxel_size
        origin     = np.asarray(voxel_grid.origin)

        all_indices   = np.array([v.grid_index for v in voxels], dtype=np.int32)
        all_positions = origin + (all_indices + 0.5) * voxel_size
        all_colors    = np.array([v.color for v in voxels])

        above = all_positions[:, 2] > z_threshold
        below = ~above

        self.positions         = all_positions[above]
        self.indices           = all_indices[above]
        self.context_positions = all_positions[below]
        self.context_colors    = all_colors[below]
        self.voxel_size        = voxel_size
        self.origin            = origin

        N = len(self.positions)
        if N == 0:
            raise ValueError(f"No voxels above z={z_threshold:.3f} m.")

        self.belief    = np.ones(N) / N
        self._has_data = False

        raw_regions = _flood_fill_xy(
            self.positions,
            cluster_radius=REGION_CLUSTER_RADIUS,
            min_voxels=REGION_MIN_VOXELS,
        )
        self.regions: List[Region] = []
        for k, member_idx in enumerate(raw_regions):
            pts = self.positions[member_idx]
            self.regions.append(Region(
                region_id=k,
                voxel_idx=member_idx,
                centre_xy=pts[:, :2].mean(axis=0),
                centre_xyz=pts.mean(axis=0),
            ))

        self._voxel_to_region = np.full(N, -1, dtype=np.int32)
        for r in self.regions:
            self._voxel_to_region[r.voxel_idx] = r.region_id

        self._ruled_out: set = set()
        self._last_check_region: Optional[int] = None
        self._last_check_xy: Optional[np.ndarray] = None
        self._checked_xys: List[np.ndarray] = []

        print(f"[BeliefFilter] {N} active voxels above z={z_threshold:.3f} m, uniform prior {1/N*100:.4f}% each.")
        print(f"[BeliefFilter] Extracted {len(self.regions)} region(s):")
        for r in self.regions:
            print(f"  {r}")
        print()

    def apply_action(self, action: Action, xy: Optional[np.ndarray] = None) -> None:
        if action == Action.NONE or xy is None:
            return
        xy = np.asarray(xy, dtype=float).reshape(2)

        if action == Action.CHECK:
            self._last_check_xy = xy.copy()
            self._last_check_region = self._nearest_region(xy)
            if self._last_check_region is not None:
                print(f"[BeliefFilter] CHECK registered at ({xy[0]*1000:.1f}, {xy[1]*1000:.1f}) mm → region {self._last_check_region}.")
            else:
                print(f"[BeliefFilter] CHECK registered at ({xy[0]*1000:.1f}, {xy[1]*1000:.1f}) mm → no region matched.")
            return

        if action == Action.PUSH:
            j = self._nearest_voxel(xy)
            self.belief[:] = 0.0
            self.belief[j] = 1.0
            self._has_data = True
            print(f"[BeliefFilter] PUSH at ({xy[0]*1000:.1f}, {xy[1]*1000:.1f}) mm → collapsed to voxel {j}.")

    def update_audio(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        mags: np.ndarray,
        spike_percentile: float = SPIKE_PERCENTILE,
        sigma_spatial: float = SIGMA_SPATIAL,
    ) -> None:
        xs   = np.asarray(xs, dtype=float)
        ys   = np.asarray(ys, dtype=float)
        mags = np.asarray(mags, dtype=float)

        threshold  = np.percentile(mags, spike_percentile)
        spike_mask = mags >= threshold
        s_xs, s_ys, s_mags = xs[spike_mask], ys[spike_mask], mags[spike_mask]

        print(f"[BeliefFilter] Audio: {spike_mask.sum()} spikes (≥{threshold:.1f} @ {spike_percentile:.0f}th pct)")

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

    def update_visual(self, found: bool) -> None:
        if self._last_check_xy is None:
            raise RuntimeError("update_visual() called with no prior apply_action(CHECK, xy).")

        check_xy = self._last_check_xy
        self._checked_xys.append(check_xy.copy())
        region_id = self._last_check_region

        if found:
            j = self._nearest_voxel(check_xy)
            self.belief[:] = 0.0
            self.belief[j] = 1.0
            print(f"[BeliefFilter] Visual: TARGET FOUND at ({check_xy[0]*1000:.1f}, {check_xy[1]*1000:.1f}) mm (region {region_id}).")
        else:
            if region_id is not None:
                r = self.regions[region_id]
                self.belief[r.voxel_idx] = 0.0
                self._ruled_out.add(region_id)
                self._normalise()
                remaining = len(self.regions) - len(self._ruled_out)
                print(f"[BeliefFilter] Visual: NOT FOUND in region {region_id} — hard-zeroed. {remaining} region(s) remaining.")
            else:
                print("[BeliefFilter] Visual: NOT FOUND, but no region associated with last CHECK.")

        self._last_check_xy = None
        self._last_check_region = None
        self._has_data = True
        self._print_belief("after visual update")

    def region_score(self, region: Region) -> float:
        return 0.0 if len(region.voxel_idx) == 0 else float(self.belief[region.voxel_idx].mean())

    def region_probability(self, region: Region) -> float:
        return 0.0 if len(region.voxel_idx) == 0 else float(self.belief[region.voxel_idx].sum())

    def ranked_regions(self) -> List[Tuple[Region, float]]:
        scored = [(r, self.region_score(r)) for r in self.regions if r.region_id not in self._ruled_out]
        scored.sort(key=lambda x: -x[1])
        return scored

    def checked_xys(self) -> List[np.ndarray]:
        return list(self._checked_xys)

    def _nearest_voxel(self, xy: np.ndarray) -> int:
        d2 = ((self.positions[:, :2] - xy) ** 2).sum(axis=1)
        return int(np.argmin(d2))

    def _nearest_region(self, xy: np.ndarray) -> Optional[int]:
        candidates = [r for r in self.regions if r.region_id not in self._ruled_out]
        if not candidates:
            return None
        dists = [np.linalg.norm(r.centre_xy - xy) for r in candidates]
        return candidates[int(np.argmin(dists))].region_id

    def _normalise(self) -> None:
        total = self.belief.sum()
        if total < 1e-12:
            active = np.ones(len(self.belief), dtype=bool)
            for rid in self._ruled_out:
                active[self.regions[rid].voxel_idx] = False
            k = active.sum()
            self.belief[:] = 0.0
            self.belief[active] = 1.0 / k if k > 0 else 1.0 / len(self.belief)
        else:
            self.belief /= total

    def _print_belief(self, tag: str = "") -> None:
        ranked = self.ranked_regions()
        active_prob = {r.region_id: self.region_probability(r) for r, _ in ranked}
        if not ranked:
            print(f"[BeliefFilter] Belief {tag}: no active regions.\n")
            return

        total_prob = sum(active_prob.values()) or 1.0
        max_score = max(score for _, score in ranked) or 1.0

        print(f"[BeliefFilter] Belief {tag}:")
        print(f"  {'rank':>4}  {'region':>6}  {'P(target)':>10}  {'mean':>8}  {'bar (mean)':<40}  centre_xy (mm)   voxels")
        print(f"  {'-'*4:>4}  {'-'*6:>6}  {'-'*10:>10}  {'-'*8:>8}  {'-'*40:<40}  {'-'*15}  {'-'*6}")
        for rank, (r, score) in enumerate(ranked):
            prob = active_prob[r.region_id]
            frac = score / max_score
            bar = '█' * int(frac * 40) + '░' * (40 - int(frac * 40))
            cx, cy = r.centre_xy * 1000
            print(f"  {rank:>4}  {r.region_id:>6}  {prob*100:>9.2f}%  {score*100:>7.3f}%  [{bar}]  ({cx:6.1f}, {cy:6.1f})  {len(r.voxel_idx):>6}")
        print(f"  active region probability mass: {total_prob*100:.2f}%  (should be ~100% after normalisation)")
        if self._ruled_out:
            print(f"  ruled out: {sorted(self._ruled_out)}")
        print()


# ── Simple visualisation helper ──────────────────────────────────────────────

def visualise_belief_simple(bf: BeliefFilter, title: str = "Belief") -> None:
    voxel_size = bf.voxel_grid.voxel_size
    positions, colours = [], []

    region_colour_map = {}
    for r in bf.regions:
        p = bf.region_probability(r)
        colour = _belief_to_rgb(min(1.0, p))
        for idx in r.voxel_idx:
            region_colour_map[int(idx)] = colour

    for i, pos in enumerate(bf.positions):
        positions.append(pos)
        colour = region_colour_map.get(i, (0.0, 0.0, 0.5))
        colours.append(colour)

    if len(bf.context_positions) > 0:
        positions.extend(bf.context_positions.tolist())
        colours.extend(bf.context_colors.tolist())

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(positions))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colours))

    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    print(f"[Visualise] {title}")
    o3d.visualization.draw_geometries([vg, axes], window_name=title, width=1280, height=720)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
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

    MIC_PORT        = "/dev/ttyACM0"
    CUBE_TAG_FAMILY = "tag36h11"
    CUBE_TAG_SIZE   = 0.0206

    cube_detector = Detector(families=CUBE_TAG_FAMILY)

    def capture_frame():
        zed = ZedCamera()
        try:
            cv_image, point_cloud = zed.get_synchronized_frame()
            K = zed.camera_intrinsic
            return cv_image, point_cloud, K
        finally:
            zed.close()

    def detect_all_tags(cv_image, camera_intrinsic, T_cam_robot):
        if cv_image.shape[2] == 4:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        gray = numpy.ascontiguousarray(gray, dtype=numpy.uint8)

        fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
        cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]

        tags = cube_detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=CUBE_TAG_SIZE,
        )

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
        best_dist = float("inf")
        best = None
        for tag_id, T_robot_cube, T_cam_cube in tag_results:
            tag_xy = T_robot_cube[:2, 3]
            if any(numpy.linalg.norm(tag_xy - cxy) < CHECKED_TAG_RADIUS for cxy in excluded_xys):
                continue
            d = numpy.linalg.norm(tag_xy - target_xy_m)
            if d < best_dist:
                best_dist = d
                best = (tag_id, T_robot_cube, T_cam_cube)
        if best is None or best_dist > max_dist:
            return None, best_dist
        return best, best_dist

    def detect_red_cube_fullframe(cv_image, red_threshold=RED_PIXEL_THRESHOLD):
        if cv_image.shape[2] == 4:
            bgr = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
        else:
            bgr = cv_image

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 80, 80]),   np.array([10, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([160, 80, 80]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(m1, m2)

        kernel = np.ones((RED_KERNEL_SIZE, RED_KERNEL_SIZE), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        red_pixels = int(cv2.countNonZero(red_mask))
        found = red_pixels > red_threshold
        return found, red_pixels

    def safe_gohome(arm):
        code, pos = arm.get_position()
        if code == 0 and pos:
            arm.set_position(pos[0], pos[1], pos[2] + 20, pos[3], pos[4], pos[5], wait=True)
        arm.move_gohome(wait=True)

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

    try:
        cv_image, point_cloud, K = capture_frame()

        T_cam_robot = get_transform_camera_robot(cv_image, K)
        if T_cam_robot is None:
            raise RuntimeError("Could not compute camera-robot transform.")

        voxel_grid = voxelize_table(point_cloud, T_cam_robot)
        if voxel_grid is None:
            raise RuntimeError("Voxel grid creation failed.")

        bf = BeliefFilter(voxel_grid)

        print("─" * 60)
        print("STEP 0b — Initial belief")
        print("─" * 60)
        visualise_belief_simple(bf, "Step 0: Uniform prior")

        print("─" * 60)
        print("STEP 1 — Audio sweep")
        print("─" * 60)

        x_mm, y_mm, _, _ = sweep_table(arm, port=MIC_PORT)
        safe_gohome(arm)

        xs   = np.array([x_mm / 1000.0])
        ys   = np.array([y_mm / 1000.0])
        mags = np.array([100.0])

        print(f"Audio source estimate: ({x_mm:.1f}, {y_mm:.1f}) mm")
        bf.update_audio(xs, ys, mags, spike_percentile=0)
        visualise_belief_simple(bf, "Step 1: After audio sweep")

        step = 2
        found_target = False
        target_tag_T_robot_cube = None

        while not found_target:
            ranked = bf.ranked_regions()
            if not ranked:
                print("No regions remain — all ruled out. Stopping.")
                break

            print("─" * 60)
            print(f"STEP {step} — Pick next cube to check")
            print("─" * 60)

            cv_image, _, K = capture_frame()

            # Keep the same transform from setup for stability
            tag_results = detect_all_tags(cv_image, K, T_cam_robot)
            if not tag_results:
                print("No AprilTags detected — stopping.")
                break

            match = None
            picked_region = None
            dist = float("inf")

            for r, score in ranked:
                prob = bf.region_probability(r)
                m, d = find_best_tag(
                    tag_results,
                    target_xy_m=r.centre_xy,
                    excluded_xys=bf.checked_xys(),
                    max_dist=TAG_MATCH_MAX_DIST,
                )
                print(f"  region {r.region_id}  P={prob*100:5.2f}%  mean={score*100:6.3f}%  nearest unchecked tag dist={d*1000:.1f} mm" +
                      ("  [skipped]" if m is None else "  [selected]" if match is None else ""))
                if m is not None and match is None:
                    match = m
                    picked_region = r
                    dist = d

            if match is None:
                print("No un-checked tag near any remaining region. Stopping.")
                break

            tag_id, T_robot_cube, _ = match
            tag_xy = T_robot_cube[:2, 3]

            print(f"  → chose region {picked_region.region_id}, tag {tag_id}  xy=({tag_xy[0]*1000:.1f}, {tag_xy[1]*1000:.1f}) mm  dist={dist*1000:.1f} mm from region centre")

            bf.apply_action(Action.CHECK, xy=tag_xy)

            do_grasp_rotate(arm, T_robot_cube, rotate_deg=180.0)
            safe_gohome(arm)
            time.sleep(RED_SETTLE_SECONDS)

            observe_cv, _, _ = capture_frame()
            found, red_pixels = detect_red_cube_fullframe(observe_cv)

            print(f"  [observe] full-frame red pixels={red_pixels}  threshold={RED_PIXEL_THRESHOLD}  → {'FOUND' if found else 'NOT FOUND'}")

            bf.update_visual(found=found)
            visualise_belief_simple(bf, f"Step {step}: {'FOUND' if found else 'empty'} @ tag {tag_id}")

            if found:
                found_target = True
                target_tag_T_robot_cube = T_robot_cube
                break

            step += 1

        if found_target and target_tag_T_robot_cube is not None:
            print("─" * 60)
            print("FINAL STEP — Push the target cube")
            print("─" * 60)

            final_xy = target_tag_T_robot_cube[:2, 3]
            bf.apply_action(Action.PUSH, xy=final_xy)

            do_push_cube(arm, target_tag_T_robot_cube, target_xy_mm=None)
            safe_gohome(arm)
            print("\n✓ Done. Target cube pushed.")
        else:
            print("\n✗ Target not found. Stopping.")

    finally:
        arm.disconnect()