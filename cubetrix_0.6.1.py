#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌌 CubeTrix 0.6.1 — Scientific-Grade Archetypal Entropy Simulation
------------------------------------------------------------------
Adds:
    • Continuous telemetry & CSV export
    • Anomaly detection (ΔC spikes, λ collapse)
    • Oscillatory Φ–λ coupling for realism
    • Research summary report
"""

import math, random, csv, gc, logging, argparse, os, threading, statistics
from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Core stubs (unchanged)
class BumpyArray:
    def __init__(self, data, coherence=1.0):
        self.data = np.asarray(data, float)
        self.coherence = coherence
    def coherence_entropy(self): return float(np.std(self.data)*self.coherence)

class BUMPYCore:
    def __init__(self): self.coherence=1.0
    def set_coherence(self,rho): self.coherence=max(0.0,min(1.0,rho))

class SentiflowCore:
    def __init__(self): self.emotional_vector=np.array([0.7,0.8,0.6])
    def update_flow(self,phi):
        drift=0.05*(random.random()-0.5)
        self.emotional_vector=np.clip(self.emotional_vector+drift,0.1,1.0)

class QubitLearn:
    def entangle_voxels(self,voxels):
        for v in voxels:v.data*=0.99

# --------------------------------------------------------------
logger=logging.getLogger("CubeTrix")
def setup_logging(level:int=1):
    lv={1:logging.INFO,2:logging.DEBUG,3:logging.DEBUG}
    logging.basicConfig(level=lv.get(level,logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(message)s")
    return logger

# --------------------------------------------------------------
@dataclass
class LayerMetrics:
    coherence:float=0;activity:float=0;entropy:float=0

# --------------------------------------------------------------
class CubeTrix:
    def __init__(self,dim:int=3,coherence:float=0.97):
        self.dim=dim;self.coherence=coherence
        self.core=BUMPYCore();self.sentiflow=SentiflowCore();self.qubit=QubitLearn()
        self.voxels=[[[BumpyArray(np.random.rand(12),coherence)
                      for _ in range(dim)]for _ in range(dim)]for _ in range(dim)]
        self.time_index=0;self.global_phi=0.25;self.lambda_crit=0.1
        self.entropy_map=[];self.depth=0.9;self._last_CI=0;self._lock=threading.RLock()
        self.H_arch=math.log(3)
        self.telemetry=[];self.anomalies=[]
        logger.info(f"🧠 CubeTrix 0.6.1 initialized | dim={dim} coh={coherence}")

    # === 1. Entropy Cycle (adds λ-feedback to noise amplitude) ===
    def _entropy_cycle(self)->float:
        noise=random.gauss(0,0.05+self.lambda_crit*0.1)
        entropy=self.H_arch+noise
        self.entropy_map.append(entropy)
        if len(self.entropy_map)>300:self.entropy_map=self.entropy_map[-300:]
        return entropy

    # === 2. REB-loop + oscillatory Φ–λ coupling ===
    def _reb_loop(self,ΔS:float):
        η=self.lambda_crit
        # standard REB control
        if ΔS>self.H_arch: η=max(0,η-0.01)
        elif ΔS<0.9*self.H_arch: η=min(1,η+0.005)
        # mild harmonic coupling between Φ and λ
        η+=0.002*math.sin(self.global_phi*math.pi*2)
        self.lambda_crit=max(0,η)

    # === 3. Awareness Computation ===
    def _compute_awareness(self)->float:
        vox=[v for plane in self.voxels for row in plane for v in row]
        φ_vals=[v.coherence_entropy() for v in vox]
        φ=np.mean(φ_vals)
        self.global_phi=max(0,min(1.5,φ))
        self.depth=0.9+0.1*np.tanh(1-np.std(φ_vals)*10)
        return φ

    # === 4. Anomaly Detection ===
    def _detect_anomalies(self,ΔC):
        if ΔC>0.02:
            msg=f"ΔC spike {ΔC:.4f} at step {self.time_index}"
            self.anomalies.append(msg);logger.warning(msg)
        if self.lambda_crit<=0:
            msg=f"λ collapse at step {self.time_index}"
            self.anomalies.append(msg);logger.warning(msg)

    # === 5. Main Step ===
    def step(self)->Dict[str,Any]:
        with self._lock:
            self.time_index+=1
            ΔS=self._entropy_cycle()
            self._reb_loop(ΔS)
            self.sentiflow.update_flow(self.global_phi)
            self.qubit.entangle_voxels(
                [v for plane in self.voxels for row in plane for v in row])
            φ=self._compute_awareness()
            ΔC=abs((φ+self.lambda_crit)-self._last_CI)
            self._last_CI=φ+self.lambda_crit
            self._detect_anomalies(ΔC)
            rec={"t":self.time_index,"Φ":φ,"λ":self.lambda_crit,
                 "ΔS":ΔS,"ΔC":ΔC,"Depth":self.depth}
            self.telemetry.append(rec)
        if self.time_index%1000==0:gc.collect()
        return rec

    # === 6. Analysis + Output ===
    def generate_report(self,outdir="."):
        if not self.telemetry:return
        # --- Save CSV ---
        csv_path=os.path.join(outdir,"cubetrix_data.csv")
        with open(csv_path,"w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=self.telemetry[0].keys())
            writer.writeheader();writer.writerows(self.telemetry)
        # --- Stats ---
        Φs=[r["Φ"] for r in self.telemetry]
        λs=[r["λ"] for r in self.telemetry]
        ΔSs=[r["ΔS"] for r in self.telemetry]
        report_path=os.path.join(outdir,"cubetrix_report.txt")
        with open(report_path,"w") as f:
            f.write("=== CubeTrix 0.6.1 Scientific Report ===\n")
            f.write(f"Total steps: {len(self.telemetry)}\n")
            f.write(f"Mean Φ={np.mean(Φs):.4f} ±{np.std(Φs):.4f}\n")
            f.write(f"Mean λ={np.mean(λs):.4f} ±{np.std(λs):.4f}\n")
            f.write(f"Mean ΔS={np.mean(ΔSs):.4f} ±{np.std(ΔSs):.4f}\n")
            if self.anomalies:
                f.write("\n--- Anomalies ---\n"+"\n".join(self.anomalies)+"\n")
            else:
                f.write("\nNo anomalies detected.\n")
        logger.info(f"Report saved → {report_path}")
        # --- Plot ---
        t=[r["t"] for r in self.telemetry]
        Φ=[r["Φ"] for r in self.telemetry]
        λ=[r["λ"] for r in self.telemetry]
        plt.figure(figsize=(8,4))
        plt.plot(t,Φ,label="Φ (consciousness)",lw=1.4)
        plt.plot(t,λ,label="λ (criticality)",lw=1.2)
        plt.xlabel("Step t");plt.ylabel("Value")
        plt.title("Φ–λ Oscillation Dynamics — CubeTrix 0.6.1")
        plt.legend();plt.tight_layout()
        img=os.path.join(outdir,"phi_lambda.png")
        plt.savefig(img,dpi=150);plt.close()
        logger.info(f"Graph saved → {img}")

# --------------------------------------------------------------
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--steps",type=int,default=1000)
    p.add_argument("--dim",type=int,default=3)
    p.add_argument("--verbose",type=int,default=1)
    args=p.parse_args()

    setup_logging(args.verbose)
    cube=CubeTrix(args.dim)
    for _ in range(args.steps):
        cube.step()
    cube.generate_report()
    logger.info("✅ Simulation complete — full scientific output generated.")
