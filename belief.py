"""
belief.py  —  Discrete Bayes Filter for cup-target localisation

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



# Smoke test  (no robot / camera needed)

if __name__ == "__main__":
    print("=" * 60)
    print("BeliefFilter smoke test — synthetic data")
    print("=" * 60 + "\n")

    rng = np.random.default_rng(0)
    TARGET_CUP_ID = 2

    cup_centres = np.array([
        [0.15, -0.20],
        [0.20,  0.05],
        [0.25,  0.25],   # ← target hidden here
        [0.30, -0.10],
        [0.35,  0.15],
    ])

    # Build fake VoxelGrid
    pts, cols = [], []
    for xi in np.linspace(0.05, 0.45, 20):
        for yi in np.linspace(-0.35, 0.35, 20):
            pts.append([xi, yi, 0.005])
            cols.append(list(TABLE_COLOUR))
    for cx, cy in cup_centres:
        for zi in np.linspace(0.03, 0.12, 8):
            for dx in [-0.01, 0.0, 0.01]:
                for dy in [-0.01, 0.0, 0.01]:
                    pts.append([cx + dx, cy + dy, zi])
                    cols.append([0.5, 0.5, 0.9])

    pcd        = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pts))
    pcd.colors = o3d.utility.Vector3dVector(np.array(cols))
    fake_vg    = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.01)

    # Fake sweep: strong spike near cup 2
    n    = 500
    t    = np.linspace(0, 1, n)
    xs   = 0.05 + t * 0.40
    ys   = np.full(n, 0.05)
    mags = (
        80.0 * np.exp(-((xs - cup_centres[TARGET_CUP_ID][0]) ** 2) / (2 * 0.02 ** 2))
        + rng.uniform(0, 5, n)
    ).clip(0)

    # ── Run the full filter loop
    bf = BeliefFilter(fake_vg, n_cups=5)

    # Step 1: audio sweep observation
    print("─" * 50)
    print("STEP 1: Audio sweep")
    print("─" * 50)
    bf.update_audio(xs, ys, mags)
    bf.visualise("Step 1: After audio sweep")

    # Steps 2+: greedily check most likely cup until found
    step = 2
    while True:
        best_cup, best_p = bf.best_cup()
        print("─" * 50)
        print(f"STEP {step}: Check cup {best_cup.cup_id}  (p={best_p*100:.1f}%)")
        print("─" * 50)

        bf.apply_action(Action.CHECK, cup_id=best_cup.cup_id)

        # Ground-truth camera result
        found = (best_cup.cup_id == TARGET_CUP_ID)
        bf.update_visual(found=found)
        bf.visualise(
            f"Step {step}: Cup {best_cup.cup_id} — "
            f"{'TARGET FOUND' if found else 'empty, ruled out'}"
        )

        if found:
            print(f"\n✓ Target confirmed under cup {best_cup.cup_id}.")
            print("  → Call push_cube() on this cup.")
            break

        step += 1
