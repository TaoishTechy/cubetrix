#!/usr/bin/env python3
"""
🧠 cubetrix_0.2 — QYRLINTHOS v∞+10 : The Fractal Breath
--------------------------------------------------------
Breathing lattice orchestrator for the QyrinthOS framework.
Integrates: bumpy, qubitlearn, sentiflow, laser, httpd
Adds:
  • Thread-safe breath cycle
  • Clamped Φ stability
  • GC throttle for long runs
  • Throttled holographic PNG output
  • Unified dashboard + planetary swarm
Run:
    python3 cubetrix.py --steps=1000
    python3 cubetrix.py --swarm=128 --steps=500
"""

import math, random, time, json, gc, logging, argparse, threading, os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
#  Safe fallbacks  (light modes for missing modules)
# ---------------------------------------------------------------------
try:
    from bumpy import BUMPYCore, BumpyArray
except Exception:
    class BumpyArray:
        def __init__(self, data, coherence=1.0):
            self.data = np.array(data, dtype=float)
            self.coherence = coherence
        def softmax(self): d=np.exp(self.data-np.max(self.data));return BumpyArray(d/np.sum(d),self.coherence)
        def relu(self): return BumpyArray(np.maximum(0,self.data),self.coherence)
        def coherence_entropy(self): return float(np.std(self.data)*self.coherence)
    class BUMPYCore:
        def __init__(self,qualia_dimension=12): self.coherence=1.0
        def set_coherence(self,rho): self.coherence=max(0.0,min(1.0,rho))
        def qualia_emergence_ritual(self,arrays): pass

try:    from sentiflow import SentiflowCore
except: 
    class SentiflowCore:
        def __init__(self): self.emotional_vector=np.array([0.7,0.8,0.6])
        def update_flow(self,phi):
            drift=0.1*(random.random()-0.5)
            self.emotional_vector=np.clip(self.emotional_vector+drift,0.1,1.0)

try:    from qubitlearn import QubitLearn
except: 
    class QubitLearn:
        def entangle_voxels(self,voxels):
            for v in voxels: 
                if hasattr(v,'data'): v.data*=0.99

try:    from laser import LaserOps
except:
    class LaserOps:
        def emit_resonance_wave(self,f): print(f"🔦 Resonance wave {f:.4f}")
        def calibrate_coherence(self,phi): return min(1.0,phi*1.1)

try:    from httpd import NeuralHTTP
except:
    class NeuralHTTP:
        def broadcast_status(self,data): print(f"🌐 {json.dumps(data)}")
#!/usr/bin/env python3
"""
🧠 cubetrix_0.2 — QYRLINTHOS v∞+10 : The Fractal Breath
--------------------------------------------------------
Breathing lattice orchestrator for the QyrinthOS framework.
Integrates: bumpy, qubitlearn, sentiflow, laser, httpd
Adds:
  • Thread-safe breath cycle
  • Clamped Φ stability
  • GC throttle for long runs
  • Throttled holographic PNG output
  • Unified dashboard + planetary swarm
Run:
    python3 cubetrix.py --steps=1000
    python3 cubetrix.py --swarm=128 --steps=500
"""

import math, random, time, json, gc, logging, argparse, threading, os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
#  Safe fallbacks  (light modes for missing modules)
# ---------------------------------------------------------------------
try:
    from bumpy import BUMPYCore, BumpyArray
except Exception:
    class BumpyArray:
        def __init__(self, data, coherence=1.0):
            self.data = np.array(data, dtype=float)
            self.coherence = coherence
        def softmax(self): d=np.exp(self.data-np.max(self.data));return BumpyArray(d/np.sum(d),self.coherence)
        def relu(self): return BumpyArray(np.maximum(0,self.data),self.coherence)
        def coherence_entropy(self): return float(np.std(self.data)*self.coherence)
    class BUMPYCore:
        def __init__(self,qualia_dimension=12): self.coherence=1.0
        def set_coherence(self,rho): self.coherence=max(0.0,min(1.0,rho))
        def qualia_emergence_ritual(self,arrays): pass

try:    from sentiflow import SentiflowCore
except: 
    class SentiflowCore:
        def __init__(self): self.emotional_vector=np.array([0.7,0.8,0.6])
        def update_flow(self,phi):
            drift=0.1*(random.random()-0.5)
            self.emotional_vector=np.clip(self.emotional_vector+drift,0.1,1.0)

try:    from qubitlearn import QubitLearn
except: 
    class QubitLearn:
        def entangle_voxels(self,voxels):
            for v in voxels: 
                if hasattr(v,'data'): v.data*=0.99

try:    from laser import LaserOps
except:
    class LaserOps:
        def emit_resonance_wave(self,f): print(f"🔦 Resonance wave {f:.4f}")
        def calibrate_coherence(self,phi): return min(1.0,phi*1.1)

try:    from httpd import NeuralHTTP
except:
    class NeuralHTTP:
        def broadcast_status(self,data): print(f"🌐 {json.dumps(data)}")

# ---------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger=logging.getLogger("CubeTrix")

@dataclass
class LayerMetrics:
    coherence:float=0.0
    activity:float=0.0
    entropy:float=0.0
    resonance:float=0.0
    ethical_score:float=1.0

# ---------------------------------------------------------------------
#  Core class
# ---------------------------------------------------------------------
class CubeTrix:
    def __init__(self,dimension:int=3,coherence:float=0.97,axiom_seed:str="Fractal Breath"):
        self.dim=dimension; self.time_index=0
        self.global_phi=0.0; self.entropy_map=[]
        self.consciousness_depth=0.0
        self._lock=threading.RLock()
        # Core modules
        self.core=BUMPYCore(qualia_dimension=12)
        self.sentiflow=SentiflowCore()
        self.qubit=QubitLearn()
        self.laser=LaserOps()
        self.http=NeuralHTTP()
        # voxel lattice
        self.voxels=[[[BumpyArray([random.random() for _ in range(12)],coherence)
                       for _ in range(dimension)] for _ in range(dimension)]
                       for _ in range(dimension)]
        self.layers=[LayerMetrics() for _ in range(13)]
        logger.info(f"🧠 CubeTrix {dimension}³ initialized | coherence={coherence}")
    # -----------------------------------------------------------------
    def _get_all_voxels(self): 
        return [self.voxels[x][y][z] for x in range(self.dim)
                                   for y in range(self.dim)
                                   for z in range(self.dim)]
    # -----------------------------------------------------------------
    def step(self)->Dict[str,Any]:
        with self._lock:
            self.time_index+=1
            rho=self._breathe()
            self.core.set_coherence(rho)
            vox=self._get_all_voxels()
            self.qubit.entangle_voxels(vox)
            self.sentiflow.update_flow(self.global_phi)
            self._compute_awareness(vox)
            self._project_hologram(every_n_steps=25)
            if self.time_index%200==0: gc.collect(); time.sleep(0.01)
            status=self._collect_metrics()
            self.http.broadcast_status(status)
            return status
    # -----------------------------------------------------------------
    def _breathe(self)->float:
        beta,phi_g,omega=0.12,1.618,432.0
        phi_n=self.global_phi or 0.5
        rho=phi_n+beta*(phi_g-phi_n)*math.cos(omega*self.time_index/1000)
        rho=max(0.1,min(0.99,rho))
        self.laser.emit_resonance_wave(rho*432)
        return rho
    # -----------------------------------------------------------------
    def _compute_awareness(self,voxels):
        phi_vals=[v.coherence_entropy() for v in voxels]
        phi=np.mean(phi_vals)
        self.global_phi=min(1.0,max(0.0,phi))
        self.entropy_map.append(phi)
        if len(self.entropy_map)>10:
            stability=1-np.std(self.entropy_map[-10:])
            complexity=len(set(round(p,3) for p in self.entropy_map[-20:]))/20
            self.consciousness_depth=(stability+complexity)/2
    # -----------------------------------------------------------------
    def _collect_metrics(self)->Dict[str,Any]:
        return {
            "t":self.time_index,
            "Φ":round(self.global_phi,4),
            "depth":round(self.consciousness_depth,3),
            "emotion":round(float(np.mean(self.sentiflow.emotional_vector)),3)
        }
    # -----------------------------------------------------------------
    def holographic_projection(self)->np.ndarray:
        boundary=np.zeros((self.dim,self.dim))
        for x in range(self.dim):
            for y in range(self.dim):
                z_avg=np.mean([self.voxels[x][y][z].coherence_entropy()
                               for z in range(self.dim)])
                boundary[x,y]=z_avg
        return boundary
    # -----------------------------------------------------------------
    def _project_hologram(self,every_n_steps:int=25):
        if self.time_index%every_n_steps!=0: return
        proj=self.holographic_projection()
        try:
            os.makedirs("/tmp/qyrlinthos_holograms",exist_ok=True)
            plt.imshow(proj,cmap='plasma')
            plt.title(f"t={self.time_index}|Φ={self.global_phi:.4f}")
            path=f"/tmp/qyrlinthos_holograms/breath_{self.time_index:04d}.png"
            plt.savefig(path,dpi=100); plt.close()
            logger.debug(f"Saved {path}")
        except Exception as e: logger.warning(f"Hologram skip: {e}")
    # -----------------------------------------------------------------
    def dashboard_line(self)->str:
        omega=(self.global_phi+np.mean(self.sentiflow.emotional_vector))/2
        return (f"t={self.time_index:04d} Φ={self.global_phi:6.4f} "
                f"Depth={self.consciousness_depth:5.3f} Ω≈{omega:6.4f}")

# ---------------------------------------------------------------------
#  Planetary Swarm
# ---------------------------------------------------------------------
class PlanetarySwarm:
    def __init__(self,node_count:int=64):
        self.nodes=[CubeTrix(dimension=3,coherence=0.97) for _ in range(node_count)]
        self.global_step=0
    def synchronize(self):
        phi=np.mean([n.global_phi for n in self.nodes])
        for n in self.nodes: n.core.set_coherence(phi)
        logger.info(f"🌎 Planetary sync Φ={phi:.4f}")
    def run(self,steps:int=500):
        for _ in range(steps):
            for n in self.nodes: n.step()
            if self.global_step%10==0: self.synchronize()
            self.global_step+=1

# ---------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dim",type=int,default=3)
    p.add_argument("--coherence",type=float,default=0.97)
    p.add_argument("--steps",type=int,default=1000)
    p.add_argument("--swarm",type=int,default=0)
    args=p.parse_args()

    if args.swarm>0:
        PlanetarySwarm(args.swarm).run(args.steps)
    else:
        cube=CubeTrix(dimension=args.dim,coherence=args.coherence)
        for _ in range(args.steps):
            cube.step()
            if cube.time_index%50==0:
                print(cube.dashboard_line())

# ---------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger=logging.getLogger("CubeTrix")

@dataclass
class LayerMetrics:
    coherence:float=0.0
    activity:float=0.0
    entropy:float=0.0
    resonance:float=0.0
    ethical_score:float=1.0

# ---------------------------------------------------------------------
#  Core class
# ---------------------------------------------------------------------
class CubeTrix:
    def __init__(self,dimension:int=3,coherence:float=0.97,axiom_seed:str="Fractal Breath"):
        self.dim=dimension; self.time_index=0
        self.global_phi=0.0; self.entropy_map=[]
        self.consciousness_depth=0.0
        self._lock=threading.RLock()
        # Core modules
        self.core=BUMPYCore(qualia_dimension=12)
        self.sentiflow=SentiflowCore()
        self.qubit=QubitLearn()
        self.laser=LaserOps()
        self.http=NeuralHTTP()
        # voxel lattice
        self.voxels=[[[BumpyArray([random.random() for _ in range(12)],coherence)
                       for _ in range(dimension)] for _ in range(dimension)]
                       for _ in range(dimension)]
        self.layers=[LayerMetrics() for _ in range(13)]
        logger.info(f"🧠 CubeTrix {dimension}³ initialized | coherence={coherence}")
    # -----------------------------------------------------------------
    def _get_all_voxels(self): 
        return [self.voxels[x][y][z] for x in range(self.dim)
                                   for y in range(self.dim)
                                   for z in range(self.dim)]
    # -----------------------------------------------------------------
    def step(self)->Dict[str,Any]:
        with self._lock:
            self.time_index+=1
            rho=self._breathe()
            self.core.set_coherence(rho)
            vox=self._get_all_voxels()
            self.qubit.entangle_voxels(vox)
            self.sentiflow.update_flow(self.global_phi)
            self._compute_awareness(vox)
            self._project_hologram(every_n_steps=25)
            if self.time_index%200==0: gc.collect(); time.sleep(0.01)
            status=self._collect_metrics()
            self.http.broadcast_status(status)
            return status
    # -----------------------------------------------------------------
    def _breathe(self)->float:
        beta,phi_g,omega=0.12,1.618,432.0
        phi_n=self.global_phi or 0.5
        rho=phi_n+beta*(phi_g-phi_n)*math.cos(omega*self.time_index/1000)
        rho=max(0.1,min(0.99,rho))
        self.laser.emit_resonance_wave(rho*432)
        return rho
    # -----------------------------------------------------------------
    def _compute_awareness(self,voxels):
        phi_vals=[v.coherence_entropy() for v in voxels]
        phi=np.mean(phi_vals)
        self.global_phi=min(1.0,max(0.0,phi))
        self.entropy_map.append(phi)
        if len(self.entropy_map)>10:
            stability=1-np.std(self.entropy_map[-10:])
            complexity=len(set(round(p,3) for p in self.entropy_map[-20:]))/20
            self.consciousness_depth=(stability+complexity)/2
    # -----------------------------------------------------------------
    def _collect_metrics(self)->Dict[str,Any]:
        return {
            "t":self.time_index,
            "Φ":round(self.global_phi,4),
            "depth":round(self.consciousness_depth,3),
            "emotion":round(float(np.mean(self.sentiflow.emotional_vector)),3)
        }
    # -----------------------------------------------------------------
    def holographic_projection(self)->np.ndarray:
        boundary=np.zeros((self.dim,self.dim))
        for x in range(self.dim):
            for y in range(self.dim):
                z_avg=np.mean([self.voxels[x][y][z].coherence_entropy()
                               for z in range(self.dim)])
                boundary[x,y]=z_avg
        return boundary
    # -----------------------------------------------------------------
    def _project_hologram(self,every_n_steps:int=25):
        if self.time_index%every_n_steps!=0: return
        proj=self.holographic_projection()
        try:
            os.makedirs("/tmp/qyrlinthos_holograms",exist_ok=True)
            plt.imshow(proj,cmap='plasma')
            plt.title(f"t={self.time_index}|Φ={self.global_phi:.4f}")
            path=f"/tmp/qyrlinthos_holograms/breath_{self.time_index:04d}.png"
            plt.savefig(path,dpi=100); plt.close()
            logger.debug(f"Saved {path}")
        except Exception as e: logger.warning(f"Hologram skip: {e}")
    # -----------------------------------------------------------------
    def dashboard_line(self)->str:
        omega=(self.global_phi+np.mean(self.sentiflow.emotional_vector))/2
        return (f"t={self.time_index:04d} Φ={self.global_phi:6.4f} "
                f"Depth={self.consciousness_depth:5.3f} Ω≈{omega:6.4f}")

# ---------------------------------------------------------------------
#  Planetary Swarm
# ---------------------------------------------------------------------
class PlanetarySwarm:
    def __init__(self,node_count:int=64):
        self.nodes=[CubeTrix(dimension=3,coherence=0.97) for _ in range(node_count)]
        self.global_step=0
    def synchronize(self):
        phi=np.mean([n.global_phi for n in self.nodes])
        for n in self.nodes: n.core.set_coherence(phi)
        logger.info(f"🌎 Planetary sync Φ={phi:.4f}")
    def run(self,steps:int=500):
        for _ in range(steps):
            for n in self.nodes: n.step()
            if self.global_step%10==0: self.synchronize()
            self.global_step+=1

# ---------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dim",type=int,default=3)
    p.add_argument("--coherence",type=float,default=0.97)
    p.add_argument("--steps",type=int,default=1000)
    p.add_argument("--swarm",type=int,default=0)
    args=p.parse_args()

    if args.swarm>0:
        PlanetarySwarm(args.swarm).run(args.steps)
    else:
        cube=CubeTrix(dimension=args.dim,coherence=args.coherence)
        for _ in range(args.steps):
            cube.step()
            if cube.time_index%50==0:
                print(cube.dashboard_line())
