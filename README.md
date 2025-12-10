# C-DPSO: A CUDA-Accelerated Chaos-Enhanced Dynamic Parameter PSO for Predictable and Rapid Multi-UAV Path Replanning

## Overview

This repository contains the source code and experimental results for the paper titled: **"C-DPSO: A CUDA-Accelerated Chaos-Enhanced Dynamic Parameter PSO for Predictable and Rapid Multi-UAV Path Replanning."**

The core contribution is the implementation and evaluation of the **Chaos-Enhanced Dynamic Parameter Particle Swarm Optimization (C-DPSO)** algorithm, optimized using **NVIDIA CUDA** via CuPy and Numba, to solve the complex, multi-objective 3D path planning and dynamic collision avoidance problem for decentralized UAV (Drone) Swarms.

---

## 🚀 Key Contributions & Results

The C-DPSO algorithm is designed to prioritize **Predictable Execution Time** and **Rapid Feasibility** in dynamic environments, a critical requirement for real-time robotic systems.

### 1. Algorithm Novelty (C-DPSO)

*   C-DPSO dynamically adjusts PSO coefficients ($W, C_1, C_2$) based on the Logistic Chaos Map, replacing static or linear parameter decay schemes.
*   The system is fully vectorized using CuPy arrays and accelerated using Numba CUDA kernels for computationally intensive tasks like collision checking.

### 2. Experimental Performance (Scenario 2: Large Swarm, Dynamic Constraints)

The C-DPSO was compared against Standard PSO (SPSO) over 10 independent runs (N=10) on a dual NVIDIA RTX 4090 system (CUDA 12.5).

| Metric | SPSO (Baseline) | C-DPSO (Proposed) | Conclusion |
| :--- | :--- | :--- | :--- |
| **Mean Fitness (Lower is Better)** | **13754.16** | 19848.37 | SPSO achieves better Global Optimum. |
| **Mean Time (s)** | 2.2675 | 2.2896 | Time is comparable. |
| **Time Std Dev (s)** | 0.0433 | **0.0061** | **Major Advantage:** C-DPSO's execution time is significantly more stable (7x more predictable). |

**Conclusion:** C-DPSO sacrifices global optimality for superior **Time Predictability**, making it robust for real-time (sub-3 second) path replanning cycles.

---

## 🛠️ Setup and Installation

This project requires a GPU with CUDA installed and uses a Conda environment for dependency management.

### Prerequisites

*   NVIDIA GPU (Tested on RTX 4090)
*   CUDA Toolkit (Tested on 12.5)
*   Conda (Anaconda/Miniconda)

### Environment Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/ailabteam/CUDA-QISO-UAV-Swarm.git
    cd CUDA-QISO-UAV-Swarm
    ```
2.  **Create and Activate Conda Environment:**
    ```bash
    conda create -n qiso_uav python=3.10
    conda activate qiso_uav
    ```
3.  **Install Dependencies:** *(Ensure CuPy version matches your CUDA toolkit version. CUDA 12.x requires `cupy-cuda12x`)*
    ```bash
    pip install numpy scipy matplotlib
    pip install cupy-cuda12x numba
    ```

---

## ⚙️ Running Experiments

The project is structured around two main scenarios defined in the `data/` directory. The main runner executes statistical analysis (N=10 runs) across both scenarios for comparison.

### Project Structure

```
CUDA-QISO-UAV-Swarm/
├── data/
│   ├── config_scenario1.py     # Static, small swarm setup (4 UAVs)
│   └── config_scenario2.py     # Dynamic, large swarm setup (8 UAVs)
├── src/
│   ├── Environment.py          # Problem formulation and objective function
│   ├── QISO_Core.py            # C-DPSO algorithm logic (SPSO/Dynamic Parameter)
│   ├── QISO_CUDA_Kernels.py    # Numba CUDA kernels for collision checks
│   └── Runner.py               # Main statistical execution script
├── results/                    # Output directory for logs and plots
└── README.md
```

### Execution

Run the statistical comparison using the Python module execution flag:

```bash
conda activate qiso_uav
python -m src.Runner
```

### Outputs

The `results/` directory will contain:

*   `statistical_summary.csv`: Table containing Mean, StdDev, Min Fitness, and Mean Time for all scenarios/algorithms.
*   `Scenario_X_Comparison_mean_convergence.pdf`: Plots comparing the mean convergence history of SPSO and C-DPSO.
*   `Scenario_X_Algorithm_best_path.pdf`: 3D visualization of the best path found during the runs.

---

## 📝 Problem Formulation Details

The optimization minimizes the multi-objective fitness function $F$:

$$F = w_1 f_{\text{Distance}} + w_2 f_{\text{Collision}} + w_3 f_{\text{Task}}$$

Where $f_{\text{Collision}}$ includes penalties for **static obstacles** and **dynamic UAV-UAV separation constraints**, heavily accelerated by custom Numba CUDA kernels.
