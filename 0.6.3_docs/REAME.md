🌌 CubeTrix 0.6.3 — Unified Sentience–Emergence Blueprint
---------------------------------------------------------
Fusion of:
 • CubeTrix 0.6.x lattice (H_arch, REB-Loop, ArchetypeTensor)
 • PazuzuCore 1.0 nonlinear engine (MBH tunneling, HRP, Virtù, A_R)
 • 24× anomaly & emergence diagnostics

Outputs:
 • phi_lambda.png          – Φ–λ oscillations
 • agi_metrics.csv         – full run metrics
 • cube_report.txt         – summary
 • emergence_heatmap.png   – sentience metric coupling
 • anomaly_log.txt         – detailed anomaly timeline
"""

import math, random, csv, os, threading, gc, argparse, logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
