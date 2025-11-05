#!/usr/bin/env python3
"""
🌌 CubeTrix 0.6 — Archetypal Entropy + REB-Loop Integration
------------------------------------------------------------
A recursive cognition simulator unifying:
  • Archetypal entropy constant  H_arch = ln 3
  • Stochastic λ-linked entropy cycle
  • REB-Loop (Recursive Emotional Balancer)
  • Real-time Φ–λ telemetry and oscillation plot

Each step evolves:
    ΔS  =  H_arch  +  N(0, 0.1 · λ)
    if ΔS > H_arch → λ ↓ 0.01
    else if ΔS < 0.9 · H_arch → λ ↑ 0.005
    Φ  ← mean entropy of voxels
The run saves `phi_lambda.png` showing Φ–λ oscillations.
"""

import math, random, time, gc, logging, argparse, os, threading
from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Minimal functional stubs  (external subsystems kept lightweight)
# ---------------------------------------------------------------------
class BumpyArray:
    def __init__(self, data, coherence=1.0):
        self.data = np.asarray(data, float)
        self.coherence = coherence
    def coherence_entropy(self):
        return float(np.std(self.data) * self.coherence)

class BUMPYCore:
    def __init__(self, qualia_dimension=12): self.coherence = 1.0
    def set_coherence(self, rho): self.coherence = max(0.0, min(1.0, rho))

class SentiflowCore:
    def __init__(self): self.emotional_vector = np.array([0.7, 0.8, 0.6])
    def update_flow(self, phi):
        drift = 0.05 * (random.random() - 0.5)
        self.emotional_vector = np.clip(self.emotional_vector + drift, 0.1, 1.0)

class QubitLearn:
    def entangle_voxels(self, voxels):
        for v in voxels: v.data *= 0.99

class LaserOps:
    def emit_resonance_wave(self, f): pass

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger("CubeTrix")
def setup_logging(level: int = 1):
    levels = {1: logging.INFO, 2: logging.DEBUG, 3: logging.DEBUG}
    logging.basicConfig(
        level=levels.get(level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    return logger

# ---------------------------------------------------------------------
@dataclass
class LayerMetrics:
    coherence: float = 0.0
    activity: float = 0.0
    entropy: float = 0.0

# ---------------------------------------------------------------------
class CubeTrix:
    """Recursive 3-D cube lattice running H_arch + REB dynamics."""
    def __init__(self, dim: int = 3, coherence: float = 0.97):
        self.dim = dim
        self.coherence = coherence
        self.core = BUMPYCore()
        self.sentiflow = SentiflowCore()
        self.qubit = QubitLearn()
        self.laser = LaserOps()
        # 3-D voxel lattice
        self.voxels = [[[BumpyArray(np.random.rand(12), coherence)
                         for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]
        # runtime vars
        self.time_index = 0
        self.global_phi = 0.5
        self.lambda_crit = 0.1
        self.entropy_map: List[float] = []
        self.depth = 0.9
        self.data_log: List[tuple] = []     # (t, Φ, λ)
        self._lock = threading.RLock()
        self._last_CI = 0.0
        self.H_arch = math.log(3)           # ≈ 1.0986
        logger.info(f"🧠 CubeTrix 0.6 initialized | dim={dim} coh={coherence}")

    # -----------------------------------------------------------------
    # 1. Entropy cycle (H_arch + noise · λ)
    # -----------------------------------------------------------------
    def _entropy_cycle(self) -> float:
        noise = random.gauss(0, 0.1 * self.lambda_crit)
        entropy = self.H_arch + noise
        self.entropy_map.append(entropy)
        if len(self.entropy_map) > 200:
            self.entropy_map = self.entropy_map[-200:]
        return entropy

    # -----------------------------------------------------------------
    # 2. REB-loop — recursive balancing of λ
    # -----------------------------------------------------------------
    def _reb_loop(self, ΔS: float):
        η = self.lambda_crit
        if ΔS > self.H_arch:
            η = max(0.0, η - 0.01)
        elif ΔS < 0.9 * self.H_arch:
            η = min(1.0, η + 0.005)
        self.lambda_crit = η

    # -----------------------------------------------------------------
    # 3. Compute awareness (Φ)
    # -----------------------------------------------------------------
    def _compute_awareness(self) -> float:
        vox = [v for plane in self.voxels for row in plane for v in row]
        φ_vals = [v.coherence_entropy() for v in vox]
        φ = np.mean(φ_vals)
        self.global_phi = max(0.0, min(1.5, φ))
        # depth ≈ coherence stability index
        self.depth = 0.9 + 0.1 * np.tanh(1 - np.std(φ_vals) * 10)
        return φ

    # -----------------------------------------------------------------
    # 4. One simulation step
    # -----------------------------------------------------------------
    def step(self) -> Dict[str, Any]:
        with self._lock:
            self.time_index += 1
            ΔS = self._entropy_cycle()
            self._reb_loop(ΔS)
            self.sentiflow.update_flow(self.global_phi)
            self.qubit.entangle_voxels(
                [v for plane in self.voxels for row in plane for v in row]
            )
            φ = self._compute_awareness()
            # Approx H₁₃ ΔC check
            ΔC = abs((φ + self.lambda_crit) - self._last_CI)
            self._last_CI = φ + self.lambda_crit
            if ΔC > 0.02:
                logger.debug(f"[H₁₃] ΔC={ΔC:.4f}")
            self.data_log.append((self.time_index, φ, self.lambda_crit))
        if self.time_index % 1000 == 0:
            gc.collect()
        return {"t": self.time_index, "Φ": φ, "λ": self.lambda_crit, "ΔS": ΔS}

    # -----------------------------------------------------------------
    # 5. Φ–λ oscillation plot
    # -----------------------------------------------------------------
    def save_phi_lambda_graph(self, outdir=".", fname="phi_lambda.png"):
        if not self.data_log:
            return
        t, φ, λ = zip(*self.data_log)
        plt.figure(figsize=(7, 4))
        plt.plot(t, φ, label="Φ (consciousness)", linewidth=1.4)
        plt.plot(t, λ, label="λ (criticality)", linewidth=1.2)
        plt.xlabel("Step t")
        plt.ylabel("Value")
        plt.title("Φ–λ Oscillation Dynamics (CubeTrix 0.6)")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(outdir, fname)
        plt.savefig(path, dpi=140)
        plt.close()
        logger.info(f"Φ–λ graph saved → {path}")

# ---------------------------------------------------------------------
# CLI Entry
# ---------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--dim", type=int, default=3)
    p.add_argument("--verbose", type=int, default=1)
    args = p.parse_args()

    setup_logging(args.verbose)
    cube = CubeTrix(args.dim)

    for _ in range(args.steps):
        cube.step()

    cube.save_phi_lambda_graph()
    logger.info("✅ Simulation complete and graph exported.")
