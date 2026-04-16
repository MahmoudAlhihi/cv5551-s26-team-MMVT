"""
belief_integrated.py  —  Per-voxel Bayes filter for shell-game target localisation
with responsive GUIs that never hang.

This version keeps the current region / tag-matching / belief / push logic,
but replaces the ROI-based red check with full-frame HSV red-pixel detection.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d


# ── Tuneable parameters ──────────────────────────────────────────────────────

VOXEL_Z_THRESHOLD   = 0.03     # m — only voxels above this get belief

SPIKE_PERCENTILE    = 85
SIGMA_SPATIAL       = 0.06     # m

# Region clustering (XY flood-fill, like the original belief.py cup extraction)
REGION_CLUSTER_RADIUS = 0.08   # m — XY radius for flood-fill neighbour linking
REGION_MIN_VOXELS     = 5      # discard tiny noise clusters

# Visual likelihood (region-level)
P_VIS_FOUND         = 0.90
P_VIS_NOT_FOUND     = 0.10

# Red detection (full-frame HSV)
RED_SETTLE_SECONDS  = 0.6
RED_PIXEL_THRESHOLD = 300
RED_KERNEL_SIZE     = 5

TAG_MATCH_MAX_DIST  = 0.10     # m — reject match if tag farther than this from region centre
CHECKED_TAG_RADIUS  = 0.04     # m — tags within this of a past check are "already done"

POLL_HZ             = 30       # display pump rate


class Action(Enum):
    NONE  = auto()
    CHECK = auto()
    PUSH  = auto()


# ── Colour helpers (used in BOTH main and viz process) ───────────────────────

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


# ── Region (flood-filled cluster of active voxels) ───────────────────────────

class Region:
    """A spatial cluster of active voxels sharing one cube/cup hypothesis."""
    __slots__ = ("region_id", "voxel_idx", "centre_xy", "centre_xyz")

    def __init__(self, region_id: int, voxel_idx: np.ndarray,
                 centre_xy: np.ndarray, centre_xyz: np.ndarray):
        self.region_id  = region_id
        self.voxel_idx  = voxel_idx
        self.centre_xy  = centre_xy
        self.centre_xyz = centre_xyz

    def __repr__(self):
        cx, cy = self.centre_xy * 1000
        return (f"Region(id={self.region_id}, "
                f"xy=({cx:.1f}, {cy:.1f}) mm, voxels={len(self.voxel_idx)})")


def _flood_fill_xy(
    positions: np.ndarray,
    cluster_radius: float,
    min_voxels: int,
) -> List[np.ndarray]:
    """
    Greedy BFS flood-fill on XY coordinates.
    Returns a list of int arrays (each the voxel indices belonging to one region).
    Regions are sorted by size, largest first.
    """
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


# ── BeliefFilter ─────────────────────────────────────────────────────────────

class BeliefFilter:
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
        all_colors    = np.array([v.color for v in voxels])

        above = all_positions[:, 2] > z_threshold
        below = ~above
        self.positions         = all_positions[above]
        self.indices           = all_indices[above]
        self.context_positions = all_positions[below]
        self.context_colors    = all_colors[below]

        self.voxel_size = voxel_size
        self.origin     = origin

        N = len(self.positions)
        if N == 0:
            raise ValueError(
                f"No voxels above z={z_threshold:.3f} m. "
                "Lower VOXEL_Z_THRESHOLD or check robot-frame Z calibration."
            )

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
                region_id  = k,
                voxel_idx  = member_idx,
                centre_xy  = pts[:, :2].mean(axis=0),
                centre_xyz = pts.mean(axis=0),
            ))

        self._voxel_to_region = np.full(N, -1, dtype=np.int32)
        for r in self.regions:
            self._voxel_to_region[r.voxel_idx] = r.region_id

        self._ruled_out: set = set()
        self._last_check_region: Optional[int] = None
        self._last_check_xy: Optional[np.ndarray] = None
        self._checked_xys: List[np.ndarray] = []

        print(f"[BeliefFilter] {N} active voxels above z={z_threshold:.3f} m, "
              f"uniform prior {1/N*100:.4f}% each.")
        print(f"[BeliefFilter] Extracted {len(self.regions)} region(s):")
        for r in self.regions:
            print(f"  {r}")
        print()

    def apply_action(self, action: Action, xy: Optional[np.ndarray] = None) -> None:
        if action == Action.NONE or xy is None:
            return
        xy = np.asarray(xy, dtype=float).reshape(2)

        if action == Action.CHECK:
            self._last_check_xy     = xy.copy()
            self._last_check_region = self._nearest_region(xy)
            if self._last_check_region is not None:
                r = self.regions[self._last_check_region]
                print(f"[BeliefFilter] CHECK registered at "
                      f"({xy[0]*1000:.1f}, {xy[1]*1000:.1f}) mm → region {r.region_id}.")
            else:
                print(f"[BeliefFilter] CHECK registered at "
                      f"({xy[0]*1000:.1f}, {xy[1]*1000:.1f}) mm → no region matched.")
            return

        if action == Action.PUSH:
            j = self._nearest_voxel(xy)
            self.belief[:] = 0.0
            self.belief[j] = 1.0
            self._has_data = True
            print(f"[BeliefFilter] PUSH at ({xy[0]*1000:.1f}, {xy[1]*1000:.1f}) mm "
                  f"→ collapsed to voxel {j}.")
            return

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

    def update_visual(self, found: bool) -> None:
        """
        Region-level visual update.

        found=True   → collapse belief to a delta on the voxel nearest check_xy.
        found=False  → HARD-zero every voxel in the checked region, add the
                       region to the ruled-out set, then renormalise.
        """
        if self._last_check_xy is None:
            raise RuntimeError(
                "update_visual() called with no prior apply_action(CHECK, xy)."
            )

        check_xy = self._last_check_xy
        self._checked_xys.append(check_xy.copy())
        region_id = self._last_check_region

        if found:
            j = self._nearest_voxel(check_xy)
            self.belief[:] = 0.0
            self.belief[j] = 1.0
            print(f"[BeliefFilter] Visual: TARGET FOUND at "
                  f"({check_xy[0]*1000:.1f}, {check_xy[1]*1000:.1f}) mm "
                  f"(region {region_id}).")
        else:
            if region_id is not None:
                r = self.regions[region_id]
                self.belief[r.voxel_idx] = 0.0
                self._ruled_out.add(region_id)
                self._normalise()
                remaining = len(self.regions) - len(self._ruled_out)
                print(f"[BeliefFilter] Visual: NOT FOUND in region {region_id} — "
                      f"hard-zeroed. {remaining} region(s) remaining.")
            else:
                print("[BeliefFilter] Visual: NOT FOUND, but no region was "
                      "associated with the last CHECK — nothing to rule out.")

        self._last_check_xy     = None
        self._last_check_region = None
        self._has_data = True
        self._print_belief("after visual update")

    def best_voxel(self) -> np.ndarray:
        j = int(np.argmax(self.belief))
        pos = self.positions[j]
        print(f"[BeliefFilter] Best voxel: idx {j}, "
              f"pos=({pos[0]*1000:.1f}, {pos[1]*1000:.1f}, {pos[2]*1000:.1f}) mm, "
              f"p={self.belief[j]*100:.2f}%")
        return pos

    def region_score(self, region: Region) -> float:
        if len(region.voxel_idx) == 0:
            return 0.0
        return float(self.belief[region.voxel_idx].mean())

    def region_probability(self, region: Region) -> float:
        if len(region.voxel_idx) == 0:
            return 0.0
        return float(self.belief[region.voxel_idx].sum())

    def region_stats(self) -> List[Tuple[Region, float, float]]:
        return [
            (r, self.region_probability(r), self.region_score(r))
            for r in self.regions
            if r.region_id not in self._ruled_out
        ]

    def ranked_regions(self) -> List[Tuple[Region, float]]:
        scored = [
            (r, self.region_score(r))
            for r in self.regions
            if r.region_id not in self._ruled_out
        ]
        scored.sort(key=lambda x: -x[1])
        return scored

    def best_region(self) -> Optional[Tuple[Region, float]]:
        ranked = self.ranked_regions()
        return ranked[0] if ranked else None

    def top_voxels(self, k: int = 5) -> List[Tuple[np.ndarray, float]]:
        order = np.argsort(-self.belief)[:k]
        return [(self.positions[i].copy(), float(self.belief[i])) for i in order]

    def checked_xys(self) -> List[np.ndarray]:
        return list(self._checked_xys)

    def _nearest_voxel(self, xy: np.ndarray) -> int:
        d2 = ((self.positions[:, :2] - xy) ** 2).sum(axis=1)
        return int(np.argmin(d2))

    def _nearest_region(self, xy: np.ndarray) -> Optional[int]:
        candidates = [r for r in self.regions if r.region_id not in self._ruled_out]
        if not candidates:
            return None
        xy = np.asarray(xy, dtype=float).reshape(2)
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
            if k > 0:
                self.belief[active] = 1.0 / k
            else:
                self.belief[:] = 1.0 / len(self.belief)
        else:
            self.belief /= total

    def _print_belief(self, tag: str = "") -> None:
        ranked      = self.ranked_regions()
        active_prob = {r.region_id: self.region_probability(r) for r, _ in ranked}
        if not ranked:
            print(f"[BeliefFilter] Belief {tag}: no active regions.\n")
            return

        total_prob = sum(active_prob.values()) or 1.0
        max_score  = max(score for _, score in ranked) or 1.0

        print(f"[BeliefFilter] Belief {tag}:")
        print(f"  {'rank':>4}  {'region':>6}  {'P(target)':>10}  "
              f"{'mean':>8}  {'bar (mean)':<40}  centre_xy (mm)   voxels")
        print(f"  {'-'*4:>4}  {'-'*6:>6}  {'-'*10:>10}  "
              f"{'-'*8:>8}  {'-'*40:<40}  {'-'*15}  {'-'*6}")
        for rank, (r, score) in enumerate(ranked):
            prob = active_prob[r.region_id]
            frac = score / max_score
            bar  = "█" * int(frac * 40) + "░" * (40 - int(frac * 40))
            cx, cy = r.centre_xy * 1000
            print(f"  {rank:>4}  {r.region_id:>6}  "
                  f"{prob*100:>9.2f}%  "
                  f"{score*100:>7.3f}%  "
                  f"[{bar}]  "
                  f"({cx:6.1f}, {cy:6.1f})  {len(r.voxel_idx):>6}")
        print(f"  active region probability mass: {total_prob*100:.2f}%"
              f"  (should be ~100% after normalisation)")
        if self._ruled_out:
            print(f"  ruled out: {sorted(self._ruled_out)}")
        print()


# ── Viz process (Open3D in a separate process) ───────────────────────────────

def _viz_process_main(queue, init_payload: dict) -> None:
    active_positions   = init_payload["active_positions"]
    context_positions  = init_payload["context_positions"]
    context_colors     = init_payload["context_colors"]
    voxel_to_region    = init_payload["voxel_to_region"]
    region_voxel_lists = init_payload["region_voxel_lists"]
    n_regions          = init_payload["n_regions"]

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Live Belief  (blue=unlikely | red=likely)",
        width=1280, height=720,
    )

    if len(context_positions) > 0:
        context_pcd = o3d.geometry.PointCloud()
        context_pcd.points = o3d.utility.Vector3dVector(context_positions)
        context_pcd.colors = o3d.utility.Vector3dVector(context_colors)
        vis.add_geometry(context_pcd)

    active_pcd = o3d.geometry.PointCloud()
    active_pcd.points = o3d.utility.Vector3dVector(active_positions)
    active_pcd.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([0.0, 0.0, 0.5]), (len(active_positions), 1))
    )
    vis.add_geometry(active_pcd)

    marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.012)
    marker.paint_uniform_color([1.0, 1.0, 1.0])
    marker.compute_vertex_normals()
    vis.add_geometry(marker)
    last_marker_pos = np.zeros(3)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(axes)

    lut_size = 256
    lut = np.array([_belief_to_rgb(i / (lut_size - 1)) for i in range(lut_size)])

    def compute_colours(belief: np.ndarray, ruled_out: np.ndarray,
                        has_data: bool) -> np.ndarray:
        N = len(belief)
        if not has_data or n_regions == 0:
            return np.tile(np.array([0.0, 0.0, 0.5]), (N, 1))

        ruled = set(int(x) for x in ruled_out)
        region_scores = np.zeros(n_regions)
        region_max    = np.zeros(n_regions)
        for rid, vox_idx in enumerate(region_voxel_lists):
            if rid in ruled or len(vox_idx) == 0:
                region_scores[rid] = 0.0
                region_max[rid]    = 0.0
            else:
                region_scores[rid] = belief[vox_idx].mean()
                region_max[rid]    = belief[vox_idx].max()

        max_score = region_scores.max()
        if max_score <= 0:
            return np.tile(np.array([0.0, 0.0, 0.5]), (N, 1))

        region_norm = region_scores / max_score
        base = np.zeros(N)
        has_region = voxel_to_region >= 0
        base[has_region] = region_norm[voxel_to_region[has_region]]

        voxel_frac = np.zeros(N)
        for rid, vox_idx in enumerate(region_voxel_lists):
            if rid in ruled or region_max[rid] <= 0:
                continue
            voxel_frac[vox_idx] = belief[vox_idx] / region_max[rid]

        modulation = 0.85 + 0.30 * voxel_frac
        heat = np.clip(base * modulation, 0.0, 1.0)

        lut_idx = np.clip((heat * (lut_size - 1)).astype(np.int32), 0, lut_size - 1)
        return lut[lut_idx]

    running = True
    while running:
        while True:
            try:
                msg = queue.get_nowait()
            except Exception:
                break
            if msg is None:
                running = False
                break
            if isinstance(msg, dict) and msg.get("kind") == "belief":
                belief    = msg["belief"]
                has_data  = msg["has_data"]
                ruled_out = msg.get("ruled_out", np.array([], dtype=np.int32))

                cols = compute_colours(belief, ruled_out, has_data)
                active_pcd.colors = o3d.utility.Vector3dVector(cols)
                vis.update_geometry(active_pcd)

                j       = int(np.argmax(belief))
                new_pos = active_positions[j]
                delta   = new_pos - last_marker_pos
                if np.linalg.norm(delta) > 1e-6:
                    marker.translate(delta)
                    last_marker_pos = new_pos.copy()
                    vis.update_geometry(marker)

        if not vis.poll_events():
            running = False
            break
        vis.update_renderer()
        time.sleep(0.016)

    try:
        vis.destroy_window()
    except Exception:
        pass


class VizProcessHandle:
    def __init__(self, bf: BeliefFilter):
        ctx = mp.get_context("spawn")
        self.queue = ctx.Queue(maxsize=16)

        region_voxel_lists = [np.ascontiguousarray(r.voxel_idx) for r in bf.regions]

        init_payload = {
            "active_positions":   np.ascontiguousarray(bf.positions),
            "context_positions":  np.ascontiguousarray(bf.context_positions),
            "context_colors":     np.ascontiguousarray(bf.context_colors),
            "voxel_to_region":    np.ascontiguousarray(bf._voxel_to_region),
            "region_voxel_lists": region_voxel_lists,
            "n_regions":          len(bf.regions),
        }
        self.proc = ctx.Process(
            target=_viz_process_main,
            args=(self.queue, init_payload),
            daemon=True,
        )
        self.proc.start()
        print(f"[Viz] Started viz process pid={self.proc.pid}")

    def send_belief(self, bf: BeliefFilter) -> None:
        try:
            self.queue.put_nowait({
                "kind":      "belief",
                "belief":    np.ascontiguousarray(bf.belief),
                "has_data":  bf._has_data,
                "ruled_out": np.array(sorted(bf._ruled_out), dtype=np.int32),
            })
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.queue.put(None, timeout=1.0)
        except Exception:
            pass
        if self.proc.is_alive():
            self.proc.join(timeout=3.0)
        if self.proc.is_alive():
            self.proc.terminate()


# ── ZED capture thread + main-thread display pump ────────────────────────────

class ZedCapture:
    """Background thread that only grabs frames into a shared slot."""

    def __init__(self, zed):
        self.zed         = zed
        self._lock       = threading.Lock()
        self._latest_cv  = None
        self._latest_pc  = None
        self._latest_ts  = 0.0
        self._stop       = threading.Event()
        self._thread     = threading.Thread(target=self._run, daemon=True)
        self._got_first  = threading.Event()

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                cv_image, point_cloud = self.zed.get_synchronized_frame()
                ts = time.monotonic()
            except Exception as e:
                print(f"[ZedCapture] frame grab failed: {e}")
                time.sleep(0.05)
                continue
            with self._lock:
                self._latest_cv = cv_image
                self._latest_pc = point_cloud
                self._latest_ts = ts
            self._got_first.set()

    def latest(self, timeout: float = 3.0):
        if not self._got_first.wait(timeout=timeout):
            raise TimeoutError("No ZED frame captured within timeout.")
        with self._lock:
            return self._latest_cv.copy(), self._latest_pc

    def latest_after(self, min_ts: float, timeout: float = 3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest_ts >= min_ts and self._latest_cv is not None:
                    return self._latest_cv.copy(), self._latest_pc
            time.sleep(0.01)
        raise TimeoutError(f"No fresh ZED frame captured after t={min_ts:.3f}.")

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


class ZedDisplay:
    """
    Main-thread display controller. Call .pump() frequently.
    """

    def __init__(self, capture: ZedCapture, window_name: str = "ZED 2 - Live"):
        import cv2
        self.cv2         = cv2
        self.capture     = capture
        self.window_name = window_name
        self._label      = ""
        self._quit       = False

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.waitKey(1)

    def set_label(self, label: str) -> None:
        self._label = label

    def pump(self) -> None:
        with self.capture._lock:
            cv_image = None if self.capture._latest_cv is None else self.capture._latest_cv.copy()
        if cv_image is None:
            self.cv2.waitKey(1)
            return

        if cv_image.shape[2] == 4:
            disp = self.cv2.cvtColor(cv_image, self.cv2.COLOR_BGRA2BGR)
        else:
            disp = cv_image.copy()

        if self._label:
            self.cv2.putText(disp, self._label, (20, 40),
                             self.cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                             (0, 255, 0), 2, self.cv2.LINE_AA)

        self.cv2.imshow(self.window_name, disp)
        if (self.cv2.waitKey(1) & 0xFF) == ord('q'):
            self._quit = True

    def quit_requested(self) -> bool:
        return self._quit

    def close(self) -> None:
        try:
            self.cv2.destroyWindow(self.window_name)
        except Exception:
            pass


def pump_sleep(display: ZedDisplay, seconds: float) -> None:
    deadline = time.time() + seconds
    dt       = 1.0 / POLL_HZ
    while time.time() < deadline:
        display.pump()
        time.sleep(dt)


def wait_for_arm(arm, display: ZedDisplay, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    dt       = 1.0 / POLL_HZ
    time.sleep(0.05)
    while time.time() < deadline:
        display.pump()
        try:
            moving = arm.get_is_moving()
        except Exception:
            moving = False
        if not moving:
            display.pump()
            time.sleep(0.1)
            return
        time.sleep(dt)
    print("[wait_for_arm] Timeout waiting for motion to finish.")


# ── Main pipeline ────────────────────────────────────────────────────────────

if __name__ == "__main__":
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
    CUBE_TAG_FAMILY = "tag36h11"
    CUBE_TAG_SIZE   = 0.0206

    calib_detector = Detector(families="tag36h11")
    cube_detector  = Detector(families=CUBE_TAG_FAMILY)

    def detect_all_tags(cv_image, camera_intrinsic, T_cam_robot, detector):
        if cv_image.shape[2] == 4:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        gray = numpy.ascontiguousarray(gray, dtype=numpy.uint8)

        fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
        cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]
        tags = detector.detect(
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
        """
        Full-frame red detection:
        BGRA/BGR -> HSV -> dual red mask -> morphology open -> count red pixels.
        """
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

    def safe_gohome(arm, display):
        code, pos = arm.get_position()
        if code == 0 and pos:
            arm.set_position(pos[0], pos[1], pos[2] + 20,
                             pos[3], pos[4], pos[5], wait=False)
            wait_for_arm(arm, display)
        arm.move_gohome(wait=False)
        wait_for_arm(arm, display)

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

    zed     = ZedCamera()
    capture = ZedCapture(zed)
    capture.start()

    display: Optional[ZedDisplay] = None
    viz: Optional[VizProcessHandle] = None

    try:
        display = ZedDisplay(capture)
        display.set_label("voxelising")

        for _ in range(30):
            display.pump()
            time.sleep(1.0 / POLL_HZ)

        cv_image, point_cloud = capture.latest()

        T_cam_robot = get_transform_camera_robot(cv_image, zed.camera_intrinsic)
        if T_cam_robot is None:
            raise RuntimeError("Could not compute camera-robot transform.")
        voxel_grid = voxelize_table(point_cloud, T_cam_robot)
        if voxel_grid is None:
            raise RuntimeError("Voxel grid creation failed.")

        bf  = BeliefFilter(voxel_grid)
        viz = VizProcessHandle(bf)
        viz.send_belief(bf)
        display.set_label("uniform prior")
        pump_sleep(display, 1.0)

        print("─" * 60)
        print("STEP 1 — Audio sweep")
        print("─" * 60)
        display.set_label("audio sweep")

        x_mm, y_mm, _, _ = sweep_table(arm, port=MIC_PORT)
        safe_gohome(arm, display)

        xs   = np.array([x_mm / 1000.0])
        ys   = np.array([y_mm / 1000.0])
        mags = np.array([100.0])
        print(f"Audio source estimate: ({x_mm:.1f}, {y_mm:.1f}) mm")

        bf.update_audio(xs, ys, mags, spike_percentile=0)
        viz.send_belief(bf)
        display.set_label("after audio sweep")
        pump_sleep(display, 1.0)

        step = 2
        found_target = False
        target_tag_T_robot_cube = None

        while not found_target:
            if display.quit_requested():
                raise KeyboardInterrupt("User pressed 'q'.")

            print("─" * 60)
            print(f"STEP {step} — Pick next cube to check")
            print("─" * 60)
            display.set_label(f"step {step}: detecting tags")
            display.pump()

            ranked = bf.ranked_regions()
            if not ranked:
                print("No regions remain — all ruled out. Stopping.")
                break

            cv_image, _ = capture.latest()
            T_cam_robot = get_transform_camera_robot(cv_image, zed.camera_intrinsic)
            if T_cam_robot is None:
                print("Lost camera-robot transform.")
                break

            tag_results = detect_all_tags(cv_image, zed.camera_intrinsic, T_cam_robot, cube_detector)
            if not tag_results:
                print("No AprilTags detected — stopping.")
                break

            match         = None
            picked_region = None
            dist          = float("inf")
            for r, score in ranked:
                prob = bf.region_probability(r)
                m, d = find_best_tag(
                    tag_results,
                    target_xy_m  = r.centre_xy,
                    excluded_xys = bf.checked_xys(),
                    max_dist     = TAG_MATCH_MAX_DIST,
                )
                print(f"  region {r.region_id}  P={prob*100:5.2f}%  "
                      f"mean={score*100:6.3f}%  "
                      f"nearest unchecked tag dist={d*1000:.1f} mm" +
                      ("  [skipped]" if m is None else "  [selected]" if match is None else ""))
                if m is not None and match is None:
                    match         = m
                    picked_region = r
                    dist          = d

            if match is None:
                print("No un-checked tag near any remaining region. Stopping.")
                break

            tag_id, T_robot_cube, _ = match
            tag_xy = T_robot_cube[:2, 3]
            print(f"  → chose region {picked_region.region_id}, tag {tag_id}  "
                  f"xy=({tag_xy[0]*1000:.1f}, {tag_xy[1]*1000:.1f}) mm  "
                  f"dist={dist*1000:.1f} mm from region centre")

            bf.apply_action(Action.CHECK, xy=tag_xy)
            viz.send_belief(bf)
            display.set_label(f"step {step}: rotating tag {tag_id}")
            display.pump()

            do_grasp_rotate(arm, T_robot_cube, rotate_deg=180.0)
            safe_gohome(arm, display)

            settle_start = time.monotonic()
            display.set_label(f"step {step}: settling")
            pump_sleep(display, RED_SETTLE_SECONDS)

            try:
                observe_cv, _ = capture.latest_after(settle_start, timeout=3.0)
            except TimeoutError:
                print("  [observe] could not get post-settle frame — skipping step.")
                break

            found, red_pixels = detect_red_cube_fullframe(observe_cv)

            print(f"  [observe] full-frame red pixels={red_pixels}  "
                  f"threshold={RED_PIXEL_THRESHOLD}  "
                  f"→ {'FOUND' if found else 'NOT FOUND'}")

            bf.update_visual(found=found)
            viz.send_belief(bf)
            display.set_label(
                f"step {step}: {'FOUND' if found else 'empty'} @ tag {tag_id}"
            )
            pump_sleep(display, 0.8)

            if found:
                found_target            = True
                target_tag_T_robot_cube = T_robot_cube
                break

            step += 1

        if found_target and target_tag_T_robot_cube is not None:
            print("─" * 60)
            print("FINAL STEP — Push the target cube")
            print("─" * 60)
            display.set_label("pushing target")
            display.pump()

            final_xy = target_tag_T_robot_cube[:2, 3]
            bf.apply_action(Action.PUSH, xy=final_xy)
            viz.send_belief(bf)

            do_push_cube(arm, target_tag_T_robot_cube, target_xy_mm=None)
            safe_gohome(arm, display)
            display.set_label("done — press 'q' to quit")
            print("\n✓ Done. Target cube pushed.")
        else:
            if display is not None:
                display.set_label("not found — press 'q' to quit")
            print("\n✗ Target not found. Stopping.")

        while display is not None and not display.quit_requested():
            display.pump()
            time.sleep(1.0 / POLL_HZ)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        try:
            arm.disconnect()
        except Exception:
            pass
        capture.stop()
        try:
            zed.close()
        except Exception:
            pass
        if viz is not None:
            viz.close()
        if display is not None:
            display.close()
        cv2.destroyAllWindows()