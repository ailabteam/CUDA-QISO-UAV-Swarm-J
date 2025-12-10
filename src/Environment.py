# src/Environment.py - Nâng cấp cho Journal (Thêm hàm CPU Evaluation)

import cupy as cp
import numpy as np # Import NumPy
from src.QISO_CUDA_Kernels import static_collision_kernel, dynamic_collision_kernel, calculate_f1_cupy, calculate_f3_cupy, THREADS_PER_BLOCK
from src.QISO_CUDA_Kernels import static_collision_kernel_cpu, dynamic_collision_kernel_cpu # Import các hàm CPU mới

class UAV_Environment:

    def __init__(self, N_uavs, N_waypoints, config, cp):
        self.cp = cp
        self.N_uavs = N_uavs
        self.N_waypoints = N_waypoints

        self.config = config
        self.params = config['sim_params']

        # Lưu trữ chướng ngại vật trên cả GPU (cp) và CPU (np)
        self.obstacles = self.cp.array(config['obstacles_data'], dtype=self.cp.float32)
        self.obstacles_np = np.array(config['obstacles_data'], dtype=np.float32)
        
        self.mission_targets = self.cp.array(config['mission_targets'], dtype=self.cp.float32)
        self.mission_targets_np = np.array(config['mission_targets'], dtype=np.float32) # Lưu trên CPU
        
        self.min_separation = self.params.get('min_separation', 0.0)

        N_particles = config['qiso_params']['N_particles']
        self.blocks = (N_particles + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    def evaluate_fitness(self, positions):
        """ Đánh giá Fitness sử dụng GPU (CuPy/CUDA) """
        N_particles = positions.shape[0]

        reshaped_pos = positions.reshape(N_particles, self.N_uavs, self.N_waypoints, 3)

        # --- Tính toán F1 (Khoảng cách) ---
        distance_cost = calculate_f1_cupy(reshaped_pos)

        # --- Tính toán F2 (Va chạm Tĩnh & Động) ---
        collision_penalty_f2 = self.cp.zeros(N_particles, dtype=self.cp.float32)
        penalty_base_value = self.params['weights'][1]

        # 2a. Va chạm Tĩnh (Static)
        obstacle_count = self.obstacles.shape[0]
        if obstacle_count > 0:
            static_collision_kernel[self.blocks, THREADS_PER_BLOCK](
                positions,
                self.obstacles,
                self.N_uavs,
                self.N_waypoints,
                obstacle_count,
                penalty_base_value * 50,
                collision_penalty_f2
            )

        # 2b. Va chạm Động (Dynamic - UAV-UAV)
        if self.min_separation > 0.0:
            min_sep_sq = self.min_separation**2
            dynamic_collision_kernel[self.blocks, THREADS_PER_BLOCK](
                positions,
                self.N_uavs,
                self.N_waypoints,
                min_sep_sq,
                penalty_base_value * 100,
                collision_penalty_f2
            )

        # --- Tính toán F3 (Nhiệm vụ) ---
        task_penalty = self.cp.zeros(N_particles, dtype=self.cp.float32) # Giả định F3 là 0 như code cũ
        # task_penalty = calculate_f3_cupy(reshaped_pos, self.mission_targets)

        # Áp dụng trọng số
        w1, _, w3 = self.params['weights']
        fitness = w1 * distance_cost + collision_penalty_f2 + w3 * task_penalty

        return fitness.get() # Trả về mảng NumPy để xử lý trên CPU

    def evaluate_fitness_cpu(self, positions_np):
        """ Đánh giá Fitness sử dụng CPU (NumPy/Numba JIT) """
        
        N_particles = positions_np.shape[0]
        
        # 1. Tính F1 (Khoảng cách) - Dùng NumPy
        reshaped_pos = positions_np.reshape(N_particles, self.N_uavs, self.N_waypoints, 3)
        diff = reshaped_pos[:, :, 1:, :] - reshaped_pos[:, :, :-1, :]
        dist_sq = np.sum(diff**2, axis=3)
        distance_cost = np.sum(np.sqrt(dist_sq), axis=(1, 2))
        
        # 2. Tính F2 (Va chạm Tĩnh & Động)
        collision_penalty_f2 = np.zeros(N_particles, dtype=np.float32)
        penalty_base_value = self.params['weights'][1]

        # 2a. Va chạm Tĩnh (Static) - Chạy trên CPU
        if self.obstacles_np.shape[0] > 0:
            static_collision_kernel_cpu(
                positions_np,
                self.obstacles_np,
                self.N_uavs,
                self.N_waypoints,
                self.obstacles_np.shape[0],
                penalty_base_value * 50,
                collision_penalty_f2
            )

        # 2b. Va chạm Động (Dynamic - UAV-UAV) - Chạy trên CPU
        if self.min_separation > 0.0:
            min_sep_sq = self.min_separation**2
            dynamic_collision_kernel_cpu(
                positions_np,
                self.N_uavs,
                self.N_waypoints,
                min_sep_sq,
                penalty_base_value * 100,
                collision_penalty_f2
            )

        # 3. Tính F3 (Nhiệm vụ)
        task_penalty = np.zeros(N_particles, dtype=np.float32) # Giả định F3 là 0
        
        # Áp dụng trọng số
        w1, _, w3 = self.params['weights']
        fitness = w1 * distance_cost + collision_penalty_f2 + w3 * task_penalty
        
        return fitness
