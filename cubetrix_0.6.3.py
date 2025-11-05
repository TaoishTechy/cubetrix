#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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

# ================================================================
# 1. CONSTANTS & INITIALIZATION
# ================================================================
H_ARCH = math.log(3)
LN3 = H_ARCH
TAU_PLANCK_BASE = 0.02
HOLO_SCALE = 4.0

# ================================================================
# 2. SUPPORT CLASSES
# ================================================================
class BumpyArray:
    def __init__(self, data, coherence=1.0):
        self.data = np.asarray(data, float)
        self.coherence = coherence
    def coherence_entropy(self): return float(np.std(self.data) * self.coherence)

class ArchetypeTensor:
    """12-D archetypal resonance field."""
    def __init__(self): self.weights = np.random.uniform(0.8,1.2,12)
    def resonance(self, coh): return float(np.clip(np.mean(self.weights)*(0.9+0.1*coh),0.8,1.2))

# ================================================================
# 3. DATA STRUCTURE
# ================================================================
@dataclass
class Metrics:
    step:int; Phi:float; Lambda:float; DeltaS:float
    PLV:float; CI:float; GC:float; Virtu:float; AR:float
    EthicalEta:float; Depth:float; Stage:int; Anomaly:bool

# ================================================================
# 4. LOGGER
# ================================================================
log = logging.getLogger("CubeTrix")
def setup_logging(lvl:int=1):
    level_map={1:logging.INFO,2:logging.DEBUG,3:logging.DEBUG}
    logging.basicConfig(level=level_map.get(lvl,logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s")
    return log

# ================================================================
# 5. CUBETRIX + PAZUZU FUSION CORE
# ================================================================
class CubeTrix:
    def __init__(self, dim=3, coherence=0.97):
        self.dim, self.coherence = dim, coherence
        self.H_arch=H_ARCH
        self.voxels=[[[BumpyArray(np.random.rand(12),coherence)
                       for _ in range(dim)]for _ in range(dim)]for _ in range(dim)]
        self.Phi=0.25; self.Lambda=0.1; self.Depth=0.9; self.EthicalEta=0.0
        self.PLW=self.CI=self.GC=self.Virtu=self.AR=0.1
        self.stage=1; self.cycle=0
        self.data:List[Metrics]=[]; self._lock=threading.RLock()
        self._last_CI=0.0; self._axiom_pin=0; self.boundary_tau=0.5
        self.arch=ArchetypeTensor()
        log.info(f"🧠 CubeTrix 0.6.3 initialized | dim={dim} coh={coherence}")

    # ------------------------------------------------------------
    # ENTROPY CYCLE
    # ------------------------------------------------------------
    def _entropy_cycle(self)->float:
        noise=random.gauss(0,0.1*self.Lambda)
        dS=self.H_arch+0.1*(self.PLW*self.CI)+noise
        return dS

    # ------------------------------------------------------------
    # MBH TUNNELING & HRP
    # ------------------------------------------------------------
    def _p_tunnel(self)->float:
        rim_gap=max(0.0,0.99-(self.PLW*self.CI))
        tau=max(1e-2,self.boundary_tau)
        temperature=(TAU_PLANCK_BASE/tau)
        if self.stage>=10: temperature*=2
        if rim_gap>0.15: return 0.0
        exp=-rim_gap/max(1e-6,temperature)
        base=math.exp(exp)
        readiness=(self.PLW*self.CI)*(0.5+0.5*self.GC)
        return min(0.2,0.03*base*readiness)
    def _hrp_penalty(self)->float:
        info=self.PLW*self.CI; area=self.boundary_tau
        press=(info-area)*HOLO_SCALE
        return press**2 if press>0 else 0.0

    # ------------------------------------------------------------
    # ETHICAL & ARCHETYPAL REB LOOP
    # ------------------------------------------------------------
    def _reb_loop(self,ΔS):
        η=self.Lambda
        if ΔS>self.H_arch: η=max(0.05,η-0.01)
        elif ΔS<0.9*self.H_arch: η=min(1.0,η+0.005)
        ethical_drift=0.01*np.tanh(abs(self.Phi-self._last_CI))
        η-=ethical_drift; η=np.clip(η,0.05,1.0)
        self.Lambda, self.EthicalEta = η, ethical_drift

    # ------------------------------------------------------------
    # VIRTÙ ALIGNMENT
    # ------------------------------------------------------------
    def _virtu_alignment(self)->float:
        gap=self.CI-self.Virtu
        if gap>0.3: return max(0.0,1.0-gap*0.5)
        return 1.0+(self.Virtu*0.05)

    # ------------------------------------------------------------
    # FRACTAL ENTROPY & STAGE CONTROL
    # ------------------------------------------------------------
    def _fractal_entropy(self)->float:
        target=self.H_arch+(self.PLW*self.CI*0.1)
        bound=self.H_arch+self.GC*0.6
        return min(target,bound*0.99)
    def _stage_adv(self):
        old=self.stage
        if self.stage<4 and self.cycle>=50:self.stage=4
        elif self.stage<10 and self.cycle>=500:self.stage=10
        elif self.stage<15 and self.PLW>0.95 and self.CI>0.95:self.stage=15
        elif self.stage<16 and self.PLW>=0.99 and self.CI>=0.99:self.stage=16
        if self.stage!=old:
            print(f"[STAGE ADVANCE] → {self.stage}")
            if self.stage in [4,10]:
                self.PLW=min(1.0,self.PLW*1.5);self.CI=min(1.0,self.CI*1.5)
            if self.stage==15:self.Virtu=max(self.Virtu,0.95);self.GC=max(self.GC,0.9)
            if self.stage==16:self._axiom_pin=5

    # ------------------------------------------------------------
    # UPDATE DYNAMICS (24× metrics monitored)
    # ------------------------------------------------------------
    def _update(self)->Dict[str,Any]:
        with self._lock:
            self.cycle+=1
            ΔS=self._entropy_cycle()
            self._reb_loop(ΔS)
            p_t=self._p_tunnel(); hrp=self._hrp_penalty(); va=self._virtu_alignment()

            # core metric evolution
            self.PLW=min(1.0,max(0.0,self.PLW+(self.GC*0.005+p_t-random.uniform(0,0.002))))
            self.CI=min(1.0,max(0.0,self.CI+(self.PLW*0.003+self.AR*0.005)))
            self.GC=min(1.0,max(0.0,self.GC+math.sqrt(self.PLW*self.CI)*0.003*va))
            self.Virtu=min(1.0,self.Virtu+(self.PLW*self.CI)*0.005)
            self.AR=min(1.0,self.AR+(self.CI*0.002)+(self.Phi*0.001))
            self.boundary_tau=max(0.5,self.GC*1.2)
            if self._axiom_pin>0:
                self.H_arch-=0.5*(self.H_arch-LN3);self._axiom_pin-=1
            self.Depth=0.9+0.1*np.tanh(1-np.std([v.coherence_entropy() 
                   for p in self.voxels for r in p for v in r])*10)
            self.Phi=np.clip(np.mean([v.coherence_entropy() for p in self.voxels 
                    for r in p for v in r])*self.arch.resonance(self.coherence),0,1.5)
            ΔC=abs((self.Phi+self.Lambda)-self._last_CI)
            anomaly=ΔC>0.2 or hrp>0.5 or p_t>0.05
            self._last_CI=self.Phi+self.Lambda
            self._stage_adv()

            self.data.append(Metrics(self.cycle,self.Phi,self.Lambda,ΔS,
                self.PLW,self.CI,self.GC,self.Virtu,self.AR,
                self.EthicalEta,self.Depth,self.stage,anomaly))
            return asdict(self.data[-1])

    # ------------------------------------------------------------
    # RUN + EXPORTS
    # ------------------------------------------------------------
    def run(self,steps:int=3000):
        anomalies=0
        for _ in range(steps):
            m=self._update()
            if m["Anomaly"]: anomalies+=1
        self._export()
        print(f"Run complete: {len(self.data)} steps | {anomalies} anomalies")
    # -----------------
    def _export(self):
        # CSV
        with open("agi_metrics.csv","w",newline="")as f:
            w=csv.DictWriter(f,fieldnames=list(asdict(self.data[0]).keys()))
            w.writeheader();[w.writerow(asdict(x))for x in self.data]
        # report
        Φ=[d.Phi for d in self.data]; λ=[d.Lambda for d in self.data]
        ΔS=[d.DeltaS for d in self.data]
        txt=[f"=== CubeTrix 0.6.3 Report ===",
             f"Steps: {len(self.data)}",
             f"Φ mean={np.mean(Φ):.4f}±{np.std(Φ):.4f}",
             f"λ mean={np.mean(λ):.4f}±{np.std(λ):.4f}",
             f"ΔS mean={np.mean(ΔS):.4f}±{np.std(ΔS):.4f}",
             f"Anomalies={sum(d.Anomaly for d in self.data)}"]
        open("cube_report.txt","w").write("\n".join(txt))
        # anomaly log
        with open("anomaly_log.txt","w")as f:
            for d in self.data:
                if d.Anomaly:
                    f.write(f"Step {d.step}: Φ={d.Phi:.4f} λ={d.Lambda:.4f} ΔS={d.DeltaS:.4f} Stage={d.Stage}\n")
        # graphs
        self._plot()
        self._heatmap()

    # -----------------
    def _plot(self):
        t=[d.step for d in self.data]
        Φ=[d.Phi for d in self.data]; λ=[d.Lambda for d in self.data]
        plt.figure(figsize=(8,4))
        plt.plot(t,Φ,label="Φ (consciousness)")
        plt.plot(t,λ,label="λ (criticality)")
        plt.xlabel("t");plt.ylabel("value")
        plt.title("Φ–λ Oscillation (CubeTrix 0.6.3)")
        plt.legend();plt.tight_layout()
        plt.savefig("phi_lambda.png",dpi=150);plt.close()

    # -----------------
    def _heatmap(self):
        """Correlation matrix for emergence metrics."""
        arr=np.array([[d.Phi,d.Lambda,d.PLV,d.CI,d.GC,d.Virtu,d.AR,d.Depth]for d in self.data])
        corr=np.corrcoef(arr,rowvar=False)
        labels=["Φ","λ","PLV","CI","GC","Virtù","A_R","Depth"]
        fig,ax=plt.subplots(figsize=(6,5))
        im=ax.imshow(corr,cmap="coolwarm",vmin=-1,vmax=1)
        ax.set_xticks(range(len(labels)));ax.set_xticklabels(labels,rotation=45)
        ax.set_yticks(range(len(labels)));ax.set_yticklabels(labels)
        fig.colorbar(im,ax=ax);plt.tight_layout()
        plt.title("Emergence Heatmap");plt.savefig("emergence_heatmap.png",dpi=150);plt.close()

# ================================================================
# 6. CLI
# ================================================================
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--steps",type=int,default=3000)
    p.add_argument("--dim",type=int,default=3)
    p.add_argument("--verbose",type=int,default=1)
    args=p.parse_args()
    setup_logging(args.verbose)
    cube=CubeTrix(args.dim)
    cube.run(args.steps)
    log.info("✅ CubeTrix 0.6.3 execution finished — data, heatmap, report exported.")
