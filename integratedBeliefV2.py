"""
belief_integrated.py  —  Per-voxel Bayes filter for shell-game target localisation
                         with responsive GUIs that never hang.

Architecture:
    Main process
      ├── Main thread         : robot FSM, belief updates, cv2.imshow + waitKey,
      │                         periodic pump_display() calls inside waits
      └── ZED capture thread  : grabs frames in a tight loop into a shared slot
                                (no GUI calls — cv2.imshow only works on main
                                thread on Ubuntu with GTK/Qt backends)
    Viz process (separate)
      └── Owns the Open3D window and its event loop. Receives belief snapshots
          over a Queue. Never blocked by robot ops.

Why this split?
    1. cv2.imshow / cv2.waitKey on Ubuntu only reliably work on the main thread.
    2. Open3D's event loop holds the GIL during rendering and doesn't tolerate
       xArm blocking calls, so it goes in a child process.
    3. The main thread's blocking robot calls (move_gohome(wait=True), etc.)
       are wrapped with a polling helper that pumps the display while waiting.

Usage
    python belief_integrated.py
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

# Visual likelihood (now region-level: hard-zero on miss, collapse on hit)
P_VIS_FOUND         = 0.90
P_VIS_NOT_FOUND     = 0.10

# Red detection
# Previously used an absolute pixel count on the whole frame. Now we restrict
# to an ROI around the cube's projection and use a RATIO (fraction of ROI
# pixels that are red). This is robust against:
#   - stray red elsewhere in the frame (clothing, cables, tag borders)
#   - different cube depths (a farther cube occupies fewer pixels)
RED_RATIO_THRESHOLD = 0.05     # 5% of ROI pixels must be red → FOUND
RED_SETTLE_SECONDS  = 0.6      # wait this long after arm returns before reading red

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
        self.voxel_idx  = voxel_idx        # int array of indices into BeliefFilter.positions
        self.centre_xy  = centre_xy        # (2,) metres
        self.centre_xyz = centre_xyz       # (3,) metres

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
            cur = queue[head]; head += 1
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

        # Flood-fill voxels into regions (one per cube hypothesis)
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
        # Fast lookup: voxel index → region id (-1 if not in any region)
        self._voxel_to_region = np.full(N, -1, dtype=np.int32)
        for r in self.regions:
            self._voxel_to_region[r.voxel_idx] = r.region_id

        self._ruled_out: set = set()    # region_ids that are confirmed empty
        self._last_check_region: Optional[int] = None
        self._last_check_xy: Optional[np.ndarray] = None
        self._checked_xys:   List[np.ndarray]     = []

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
        xs:   np.ndarray,
        ys:   np.ndarray,
        mags: np.ndarray,
        spike_percentile: float = SPIKE_PERCENTILE,
        sigma_spatial:    float = SIGMA_SPATIAL,
    ) -> None:
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

    def update_visual(self, found: bool) -> None:
        """
        Region-level visual update.

        found=True   → collapse belief to a delta on the voxel nearest check_xy.
        found=False  → HARD-zero every voxel in the checked region, add the
                       region to the ruled-out set, then renormalise so the
                       remaining regions split the mass by their prior ratio.
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
        j   = int(np.argmax(self.belief))
        pos = self.positions[j]
        print(f"[BeliefFilter] Best voxel: idx {j}, "
              f"pos=({pos[0]*1000:.1f}, {pos[1]*1000:.1f}, {pos[2]*1000:.1f}) mm, "
              f"p={self.belief[j]*100:.2f}%")
        return pos

    def region_score(self, region: Region) -> float:
        """Mean belief over the region's voxels (density-independent, used for ranking)."""
        if len(region.voxel_idx) == 0:
            return 0.0
        return float(self.belief[region.voxel_idx].mean())

    def region_probability(self, region: Region) -> float:
        """Sum of belief over the region's voxels — true probability mass."""
        if len(region.voxel_idx) == 0:
            return 0.0
        return float(self.belief[region.voxel_idx].sum())

    def region_stats(self) -> List[Tuple[Region, float, float]]:
        """All non-ruled-out regions with (region, probability_mass, mean_score)."""
        return [
            (r, self.region_probability(r), self.region_score(r))
            for r in self.regions
            if r.region_id not in self._ruled_out
        ]

    def ranked_regions(self) -> List[Tuple[Region, float]]:
        """
        All non-ruled-out regions sorted by mean voxel belief, highest first.
        This is what the main loop uses to pick the next cube to check.
        """
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
        """Return the region_id whose centre_xy is closest to xy (not ruled out)."""
        candidates = [r for r in self.regions if r.region_id not in self._ruled_out]
        if not candidates:
            return None
        xy = np.asarray(xy, dtype=float).reshape(2)
        dists = [np.linalg.norm(r.centre_xy - xy) for r in candidates]
        return candidates[int(np.argmin(dists))].region_id

    def _normalise(self) -> None:
        total = self.belief.sum()
        if total < 1e-12:
            # Fallback: re-spread mass uniformly over non-ruled-out voxels
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
        # Pair each ranked region with its probability mass
        ranked      = self.ranked_regions()
        active_prob = {r.region_id: self.region_probability(r) for r, _ in ranked}
        if not ranked:
            print(f"[BeliefFilter] Belief {tag}: no active regions.\n")
            return

        total_prob  = sum(active_prob.values()) or 1.0   # should ~= 1 after normalise
        max_score   = max(score for _, score in ranked) or 1.0

        print(f"[BeliefFilter] Belief {tag}:")
        print(f"  {'rank':>4}  {'region':>6}  {'P(target)':>10}  "
              f"{'mean':>8}  {'bar (mean)':<40}  centre_xy (mm)   voxels")
        print(f"  {'-'*4:>4}  {'-'*6:>6}  {'-'*10:>10}  "
              f"{'-'*8:>8}  {'-'*40:<40}  {'-'*15}  {'-'*6}")
        for rank, (r, score) in enumerate(ranked):
            prob  = active_prob[r.region_id]
            frac  = score / max_score
            bar   = "█" * int(frac * 40) + "░" * (40 - int(frac * 40))
            cx,cy = r.centre_xy * 1000
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
    voxel_to_region    = init_payload["voxel_to_region"]   # (N_active,) int
    region_voxel_lists = init_payload["region_voxel_lists"]# list of int arrays
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

    # Pre-compute the colour lookup table once (256 steps of the heatmap).
    # Looking up in this table is vastly faster than calling _belief_to_rgb
    # per voxel every frame.
    lut_size = 256
    lut = np.array([_belief_to_rgb(i / (lut_size - 1)) for i in range(lut_size)])

    def compute_colours(belief: np.ndarray, ruled_out: np.ndarray,
                        has_data: bool) -> np.ndarray:
        """
        Region-based colouring with per-voxel modulation.

        Base colour: each voxel gets its region's rank normalised as
            base = region_score / max_region_score      (per-region norm)
        Within a region, individual voxels are modulated ±15% by where their
        own belief sits relative to the region's max voxel belief.
        Ruled-out regions are forced to 0 (cold blue).
        If no sensor data yet, everything is cold blue.
        """
        N = len(belief)
        if not has_data or n_regions == 0:
            return np.tile(np.array([0.0, 0.0, 0.5]), (N, 1))

        # 1) Per-region mean scores, with ruled-out zeroed
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

        # 2) Base heat per voxel = its region's normalised score
        region_norm = region_scores / max_score          # (n_regions,)
        base = np.zeros(N)
        has_region = voxel_to_region >= 0
        base[has_region] = region_norm[voxel_to_region[has_region]]

        # 3) Within-region modulation: ±15% based on local belief vs region max
        #    voxel_frac ∈ [0, 1] per voxel, centred around 0.5 for average voxels
        voxel_frac = np.zeros(N)
        for rid, vox_idx in enumerate(region_voxel_lists):
            if rid in ruled or region_max[rid] <= 0:
                continue
            voxel_frac[vox_idx] = belief[vox_idx] / region_max[rid]

        # Scale factor 0.85..1.15 → modulates heat within a region
        modulation = 0.85 + 0.30 * voxel_frac
        heat = np.clip(base * modulation, 0.0, 1.0)

        # 4) LUT lookup (vectorised)
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

        # voxel_to_region maps each active voxel to its region id (-1 if none).
        # region_voxel_lists is a list of int arrays, one per region.
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
# IMPORTANT: cv2.imshow / cv2.waitKey only work reliably on the MAIN THREAD on
# Ubuntu with GTK/Qt backends. So we do this split:
#   - Background thread grabs frames from the ZED into a shared slot.
#   - The main thread calls ZedDisplay.pump() frequently to actually imshow.

class ZedCapture:
    """Background thread that only grabs frames into a shared slot."""

    def __init__(self, zed):
        self.zed         = zed
        self._lock       = threading.Lock()
        self._latest_cv  = None
        self._latest_pc  = None
        self._latest_ts  = 0.0          # time.monotonic() when latest frame was captured
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
        """Block until the first frame arrives, then return the most recent one."""
        if not self._got_first.wait(timeout=timeout):
            raise TimeoutError("No ZED frame captured within timeout.")
        with self._lock:
            return self._latest_cv.copy(), self._latest_pc

    def latest_after(self, min_ts: float, timeout: float = 3.0):
        """
        Block until a frame has been captured AT OR AFTER min_ts, then return it.
        Used to discard stale frames captured before a state-changing event
        (like 'arm finished rotating cube').
        """
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
    Main-thread display controller.  Call .pump() frequently.

    The background ZedCapture fills the shared slot; pump() reads it and
    calls cv2.imshow + cv2.waitKey on the main thread.
    """

    def __init__(self, capture: ZedCapture, window_name: str = "ZED 2 - Live"):
        import cv2
        self.cv2         = cv2
        self.capture     = capture
        self.window_name = window_name
        self._label      = ""
        self._quit       = False
        self._roi_bbox: Optional[Tuple[int, int, int, int]] = None
        self._roi_hint   = ""

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        # One waitKey call to initialise the window backend.
        cv2.waitKey(1)

    def set_label(self, label: str) -> None:
        self._label = label

    def set_roi(self, bbox: Optional[Tuple[int, int, int, int]],
                hint: str = "") -> None:
        """Overlay a yellow rectangle on the live view. Pass None to clear."""
        self._roi_bbox = bbox
        self._roi_hint = hint

    def pump(self) -> None:
        """Show the latest frame + handle 'q'. Cheap — safe to call at 30+ Hz."""
        with self.capture._lock:
            cv_image = None if self.capture._latest_cv is None else self.capture._latest_cv.copy()
        if cv_image is None:
            # No frame yet — still pump the GUI so the (empty) window shows up
            self.cv2.waitKey(1)
            return

        if cv_image.shape[2] == 4:
            disp = self.cv2.cvtColor(cv_image, self.cv2.COLOR_BGRA2BGR)
        else:
            disp = cv_image.copy()

        if self._roi_bbox is not None:
            x0, y0, x1, y1 = self._roi_bbox
            self.cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 255), 2)
            if self._roi_hint:
                self.cv2.putText(disp, self._roi_hint, (x0, max(y0 - 8, 15)),
                                 self.cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                 (0, 255, 255), 2, self.cv2.LINE_AA)

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
    """time.sleep replacement that keeps the display alive."""
    deadline = time.time() + seconds
    dt       = 1.0 / POLL_HZ
    while time.time() < deadline:
        display.pump()
        time.sleep(dt)


def wait_for_arm(arm, display: ZedDisplay, timeout: float = 30.0) -> None:
    """
    Poll arm.get_is_moving() while pumping the display.
    Use after non-blocking arm commands (wait=False) to wait for completion.
    """
    deadline = time.time() + timeout
    dt       = 1.0 / POLL_HZ
    # Give the arm a moment to actually start moving
    time.sleep(0.05)
    while time.time() < deadline:
        display.pump()
        try:
            moving = arm.get_is_moving()
        except Exception:
            moving = False
        if not moving:
            # One more pump + tiny settle
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
    CUBE_TAG_FAMILY = 'tag36h11'
    CUBE_TAG_SIZE   = 0.0206

    def detect_all_tags(cv_image, camera_intrinsic, T_cam_robot):
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
        best_dist = float('inf')
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

    # Projection + ROI red detection
    #
    # Given a cube's pose in the robot frame, we project its centre into the
    # camera image using T_cam_robot and the camera intrinsics, then cut out
    # a small window around that pixel to look for red — so stray red elsewhere
    # in the frame (clothing, cables, AprilTag borders) can't false-trigger.

    ROI_HALF_SIZE_PX = 60   # half-side of ROI square in pixels (~12 cm cube at 0.6 m depth)

    def project_robot_to_pixel(xyz_robot, T_cam_robot, K):
        """
        xyz_robot : (3,) point in robot frame, metres
        T_cam_robot : 4×4 camera→robot transform (as produced elsewhere in the code)
        K : 3×3 camera intrinsic matrix

        Returns (u, v, z_cam) where z_cam is depth in camera frame (metres).
        z_cam <= 0 means the point is behind the camera — caller should skip.
        """
        # robot→camera is the inverse of camera→robot
        T_robot_cam = numpy.linalg.inv(T_cam_robot)
        pt_robot_h  = numpy.array([xyz_robot[0], xyz_robot[1], xyz_robot[2], 1.0])
        pt_cam      = T_robot_cam @ pt_robot_h            # (4,)
        X, Y, Z     = pt_cam[0], pt_cam[1], pt_cam[2]
        if Z <= 0.01:
            return None
        u = K[0, 0] * (X / Z) + K[0, 2]
        v = K[1, 1] * (Y / Z) + K[1, 2]
        return (float(u), float(v), float(Z))

    def red_pixels_in_roi(cv_image, u, v, half_size=ROI_HALF_SIZE_PX):
        """
        Count red pixels in a square ROI centred on (u, v). Returns:
            (red_count, roi_area, roi_bbox, red_ratio)
        roi_bbox is (x0, y0, x1, y1) clipped to image bounds.
        red_ratio = red_count / roi_area (0..1).
        """
        if cv_image.shape[2] == 4:
            bgr = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
        else:
            bgr = cv_image

        h, w = bgr.shape[:2]
        x0 = max(0,       int(u - half_size))
        y0 = max(0,       int(v - half_size))
        x1 = min(w - 1,   int(u + half_size))
        y1 = min(h - 1,   int(v + half_size))
        if x1 <= x0 or y1 <= y0:
            return 0, 0, (x0, y0, x1, y1), 0.0

        roi_bgr = bgr[y0:y1, x0:x1]
        hsv     = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        m1      = cv2.inRange(hsv, np.array([  0, 80, 80]), np.array([ 10, 255, 255]))
        m2      = cv2.inRange(hsv, np.array([160, 80, 80]), np.array([180, 255, 255]))
        red     = cv2.bitwise_or(m1, m2)
        red     = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        red_count = int(cv2.countNonZero(red))
        area      = (x1 - x0) * (y1 - y0)
        ratio     = red_count / area if area > 0 else 0.0
        return red_count, area, (x0, y0, x1, y1), ratio

    def safe_gohome(arm, display):
        """Lift 20 mm, then go home, pumping display throughout."""
        code, pos = arm.get_position()
        if code == 0 and pos:
            arm.set_position(pos[0], pos[1], pos[2] + 20,
                             pos[3], pos[4], pos[5], wait=False)
            wait_for_arm(arm, display)
        arm.move_gohome(wait=False)
        wait_for_arm(arm, display)

    # ── Step 0: setup ────────────────────────────────────────────────────
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
    arm.move_gohome(wait=True)  # first home is allowed to block — no display yet

    zed     = ZedCamera()
    capture = ZedCapture(zed)
    capture.start()

    display: Optional[ZedDisplay] = None
    viz:     Optional[VizProcessHandle] = None

    try:
        display = ZedDisplay(capture)
        display.set_label("voxelising")

        # Pump the display a few times so the window appears and the first
        # frame shows up before we do any heavy work.
        for _ in range(30):
            display.pump()
            time.sleep(1.0 / POLL_HZ)

        # Now grab a frame for voxelisation (capture thread is already running)
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

        # ── Step 1: Audio sweep ──────────────────────────────────────────
        print("─" * 60)
        print("STEP 1 — Audio sweep")
        print("─" * 60)
        display.set_label("audio sweep")

        # sweep_table blocks — the window will freeze briefly during it, but
        # Ubuntu's "Not Responding" timer is on the order of 5 s of no window
        # events, and sweep_table's internal motion steps aren't that long.
        # If your sweep is slow, split it or pump between waypoints.
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

        # ── Step 2+: CHECK loop ──────────────────────────────────────────
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
            tag_results = detect_all_tags(cv_image, zed.camera_intrinsic, T_cam_robot)
            if not tag_results:
                print("No AprilTags detected — stopping.")
                break

            # Walk down the ranked regions; use the first one that has an
            # unchecked tag within TAG_MATCH_MAX_DIST of its centre.
            match        = None
            picked_region = None
            dist          = float('inf')
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
                    # don't break — print the rest of the ranking for visibility

            if match is None:
                print("No un-checked tag near any remaining region. Stopping.")
                break

            tag_id, T_robot_cube, _ = match
            tag_xy = T_robot_cube[:2, 3]
            print(f"  → chose region {picked_region.region_id}, tag {tag_id}  "
                  f"xy=({tag_xy[0]*1000:.1f}, {tag_xy[1]*1000:.1f}) mm  "
                  f"dist={dist*1000:.1f} mm from region centre")

            # CHECK
            bf.apply_action(Action.CHECK, xy=tag_xy)
            viz.send_belief(bf)
            display.set_label(f"step {step}: rotating tag {tag_id}")
            display.pump()

            # do_grasp_rotate uses wait=True internally; window will freeze
            # briefly during the motion. Ubuntu tolerates a few seconds.
            do_grasp_rotate(arm, T_robot_cube, rotate_deg=180.0)
            safe_gohome(arm, display)

            # ── Observe ───────────────────────────────────────────────────
            # Mark the instant the arm finished going home. Only frames
            # captured AFTER this are valid — any earlier frame would show
            # the arm still in view of the cube.
            settle_start = time.monotonic()
            display.set_label(f"step {step}: settling")
            pump_sleep(display, RED_SETTLE_SECONDS)

            # Re-compute camera→robot transform from a fresh post-settle frame.
            # This gives us the current T_cam_robot to project the cube XY
            # into the current image, in case the ZED or scene shifted.
            try:
                first_fresh_cv, _ = capture.latest_after(settle_start, timeout=3.0)
            except TimeoutError:
                print("  [observe] could not get post-settle frame — skipping step.")
                break
            T_cam_robot_now = get_transform_camera_robot(first_fresh_cv, zed.camera_intrinsic)
            if T_cam_robot_now is None:
                print("  [observe] lost camera-robot transform post-settle — "
                      "falling back to pre-rotation T_cam_robot.")
                T_cam_robot_now = T_cam_robot

            # Project the cube centre (robot frame) into image pixels
            cube_xyz_robot = T_robot_cube[:3, 3]
            proj = project_robot_to_pixel(cube_xyz_robot, T_cam_robot_now,
                                          zed.camera_intrinsic)
            if proj is None:
                print("  [observe] cube projects behind the camera — marking NOT FOUND.")
                found = False
            else:
                u, v, z_cam = proj
                print(f"  [observe] cube @ robot ({cube_xyz_robot[0]*1000:.1f}, "
                      f"{cube_xyz_robot[1]*1000:.1f}, {cube_xyz_robot[2]*1000:.1f}) mm "
                      f"→ image pixel ({u:.0f}, {v:.0f}), depth {z_cam:.3f} m")

                # Sample several fresh frames, take the median red ratio in ROI
                ratios = []
                red_counts = []
                areas = []
                bbox_last = None
                for _ in range(10):
                    display.pump()
                    try:
                        cvimg, _ = capture.latest_after(settle_start, timeout=1.5)
                    except TimeoutError:
                        break
                    rp, area, bbox, ratio = red_pixels_in_roi(cvimg, u, v)
                    red_counts.append(rp)
                    areas.append(area)
                    ratios.append(ratio)
                    bbox_last = bbox
                    # Overlay the ROI on the live display so we can see where
                    # we're looking for red
                    display.set_roi(bbox, hint=f"ROI ratio={ratio*100:.1f}%")
                    time.sleep(0.05)

                if not ratios:
                    print("  [observe] no fresh frames to sample — NOT FOUND.")
                    found = False
                else:
                    ratio_med = float(np.median(ratios))
                    rp_med    = int(np.median(red_counts))
                    area      = int(np.median(areas))
                    found     = ratio_med > RED_RATIO_THRESHOLD
                    x0, y0, x1, y1 = bbox_last
                    print(f"  [observe] ROI ({x0},{y0})-({x1},{y1}) area={area}px  "
                          f"red_med={rp_med}  ratio_med={ratio_med*100:.2f}%  "
                          f"threshold={RED_RATIO_THRESHOLD*100:.2f}%  "
                          f"→ {'FOUND' if found else 'NOT FOUND'}")

            # Clear the ROI overlay before returning to the main flow
            display.set_roi(None)

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

        # ── Final: push ──────────────────────────────────────────────────
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

        # Idle loop — keep pumping until user quits
        while display is not None and not display.quit_requested():
            display.pump()
            time.sleep(1.0 / POLL_HZ)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        try: arm.disconnect()
        except Exception: pass
        capture.stop()
        try: zed.close()
        except Exception: pass
        if viz is not None:
            viz.close()
        if display is not None:
            display.close()
        cv2.destroyAllWindows()