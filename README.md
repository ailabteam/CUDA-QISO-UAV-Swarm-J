# C-DPSO: Chaos-Accelerated Dynamic Parameter PSO for Predictable and Rapid Multi-UAV Path Planning

This repository contains the source code and comprehensive experimental results for the paper titled: **"C-DPSO: Chaos-Accelerated Dynamic Parameter PSO for Predictable and Rapid Path Planning in Highly Constrained Multi-UAV Swarms."**

The research focuses on balancing solution quality, execution speed, and critically, **temporal predictability** ($\sigma$ Time) for decentralized UAV swarm path replanning in dynamic 3D environments.

## 🚀 Key Contributions & Performance Analysis

We compare three core algorithms: Standard PSO (SPSO), Linear Dynamic PSO (L-DPSO), and the proposed Chaos-Enhanced Dynamic PSO (C-DPSO).

The experiments use a large-scale scenario (8 UAVs, 15 Waypoints, Dynamic Collision Constraints, 700 Iterations).

| Metric | SPSO (Fixed) | L-DPSO (Linear Dyn.) | **C-DPSO (Chaos Dyn.)** | Primary Strength |
| :--- | :--- | :--- | :--- | :--- |
| **Fitness Mean ($\mu$ $\downarrow$)** | 25209.06 | **18014.50** | 21567.89 | L-DPSO (Best Global Optimum) |
| **Time Mean ($\mu$, s $\downarrow$)** | 2.2897 | 2.2608 | **2.1442** | **C-DPSO (Fastest Execution)** |
| **Time Std Dev ($\sigma$, s $\downarrow$)** | 0.0142 | **0.0062** | 0.0319 | L-DPSO (Highest Temporal Predictability) |
| **Fitness Std Dev ($\sigma$ $\downarrow$)** | 2154.23 | 1152.82 | **624.26** | **C-DPSO (Highest Quality Consistency)** |

**Conclusion:** The choice of parameter modulation dictates performance trade-offs: L-DPSO provides superior temporal reliability, while C-DPSO offers the fastest execution time and the most consistent solution quality, making it ideal for rapid feasibility finding.

## 💻 HPC Acceleration Details

The computationally intensive $O(N^2 M)$ multi-objective fitness evaluation is accelerated using Numba CUDA kernels and CuPy array handling.

| Metric | CPU (Numba JIT) | GPU (CUDA/Numba) | Speedup Factor |
| :--- | :--- | :--- | :--- |
| Mean Evaluation Time ($\mu$) | 3.2620 ms | 1.6517 ms | **~1.97X** |

*(Note: The modest speedup factor is attributed to GPU under-utilization due to the limited particle population size (P=500) relative to GPU capacity, a scenario common in decentralized real-time systems.)*

## 🛠️ Repository Structure and Usage

**Project Structure:**
```
CUDA-QISO-UAV-Swarm-J/
├── data/                    # Configuration files (Scenario 1 & 2)
├── src/
│   ├── Environment.py       # Problem formulation, F(X) evaluation (CPU/GPU)
│   ├── QISO_CUDA_Kernels.py # Numba CUDA kernels (Collision checks)
│   ├── QISO_Optimizer.py    # Core PSO logic (SPSO, L-DPSO, C-DPSO)
│   └── Runner.py            # Main statistical execution script
└── results/                 # Output PDFs and CSV Tables (Table 1, 2, 3)
```

**Execution:**
1. Setup environment (CuPy, Numba, NumPy).
2. Run statistical comparison:
   ```bash
   python -m src.Runner
   ```

**Outputs:**
The `results/` folder contains CSV tables of statistical data and PDF figures visualizing convergence (Figure 1), path dynamics (Figure 2), execution time distribution (Figure 3), and parameter dynamics (Figure 4).
