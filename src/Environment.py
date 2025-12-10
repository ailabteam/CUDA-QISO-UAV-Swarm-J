# src/Environment.py

import cupy as cp
from src.QISO_CUDA_Kernels import static_collision_kernel, dynamic_collision_kernel, calculate_f1_cupy, calculate_f3_cupy, THREADS_PER_BLOCK

class UAV_Environment:
    
    def __init__(self, N_uavs, N_waypoints, config, cp):
        self.cp = cp
        self.N_uavs = N_uavs
        self.N_waypoints = N_waypoints
        
        self.config = config
        self.params = config['sim_params']
        
        self.obstacles = self.cp.array(config['obstacles_data'], dtype=self.cp.float32) 
        self.mission_targets = self.cp.array(config['mission_targets'], dtype=self.cp.float32)
        self.min_separation = self.params.get('min_separation', 0.0)
        
        N_particles = config['qiso_params']['N_particles']
        self.blocks = (N_particles + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        
    def evaluate_fitness(self, positions):
        N_particles = positions.shape[0]
        
        reshaped_pos = positions.reshape(N_particles, self.N_uavs, self.N_waypoints, 3)
        
        # --- Tính toán F1 (Quãng đường/Năng lượng) ---
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
                penalty_base_value * 50, # Penalty cho tĩnh
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
                penalty_base_value * 100, # Penalty cao hơn cho động
                collision_penalty_f2
            )
        
        # --- Tính toán F3 (Nhiệm vụ) ---
        task_penalty = calculate_f3_cupy(reshaped_pos, self.mission_targets)
        
        # Áp dụng trọng số
        w1, _, w3 = self.params['weights']
        fitness = w1 * distance_cost + collision_penalty_f2 + w3 * task_penalty
        
        return fitness.get()
