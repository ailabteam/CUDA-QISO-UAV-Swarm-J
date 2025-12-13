# Optimality vs. Predictability: A Statistical and CUDA-Accelerated Analysis of Dynamic PSO for Real-Time Multi-UAV Path Planning

This repository contains the source code and comprehensive experimental results for the paper titled: "Optimality vs. Predictability: A Statistical and CUDA-Accelerated Analysis of Dynamic PSO for Real-Time Multi-UAV Path Planning."

The research focuses on balancing solution quality, execution speed, and critically, temporal predictability ($\sigma$ Time) for decentralized UAV swarm path replanning in dynamic 3D environments.

---

## 🚀 Key Contributions & Performance Analysis (Updated Results)

We compare three core algorithms: Standard PSO (SPSO), Linear Dynamic PSO (L-DPSO), and the proposed Chaos-Enhanced Dynamic PSO (C-DPSO).

The experiments use a large-scale scenario (8 UAVs, 15 Waypoints, Dynamic Collision Constraints, 700 Iterations).

| Metric | SPSO (Fixed) | L-DPSO (Linear Dyn.) | **C-DPSO (Chaos Dyn.)** | Primary Strength |
| :--- | :--- | :--- | :--- | :--- |
| Fitness Mean ($\mu \downarrow$) | 20681.33 | 15920.62 | **12279.99** | **C-DPSO (Best Global Optimum)** |
| Fitness Std Dev ($\sigma \downarrow$) | 1637.85 | 1053.67 | **628.89** | **C-DPSO (Highest Quality Consistency)** |
| Time Mean ($\mu$, s $\downarrow$) | 2.2292 | **2.1808** | 2.2148 | L-DPSO (Fastest Execution) |
| Time Std Dev ($\sigma$, s $\downarrow$) | 0.0709 | **0.0114** | 0.0175 | **L-DPSO (Highest Temporal Predictability)** |

**Conclusion:** The choice of parameter modulation dictates performance trade-offs: **C-DPSO** provides superior optimality and consistency, ideal for maximizing mission efficiency. **L-DPSO** offers superior temporal reliability, making it the preferred choice when strict hard real-time deadlines must be guaranteed.

---

## 💻 HPC Acceleration Details (Updated Results)

The computationally intensive $O(N^2 M)$ multi-objective fitness evaluation is accelerated using Numba CUDA kernels and CuPy array handling.

| Metric | CPU (Numba JIT) | GPU (CUDA/Numba) | Speedup Factor |
| :--- | :--- | :--- | :--- |
| Mean Evaluation Time ($\mu$) | 3.3939 ms | 1.5792 ms | **~2.15X** |
| Std Dev Evaluation Time ($\sigma$) | 0.2878 ms | 0.0617 ms | |

(Note: The modest speedup factor is attributed to GPU under-utilization due to the limited particle population size (P=500) relative to GPU capacity, a scenario common in decentralized real-time systems.)

---

## 🛠️ Repository Structure and Usage

### Project Structure:

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

### Execution:

1.  Setup environment (CuPy, Numba, NumPy).
2.  Run statistical comparison:
    ```bash
    python -m src.Runner
    ```
3.  **Outputs:** The `results/` folder contains CSV tables of statistical data and PDF figures visualizing convergence (Figure 1), path dynamics (Figure 2), execution time distribution (Figure 3), and parameter dynamics (Figure 4).
