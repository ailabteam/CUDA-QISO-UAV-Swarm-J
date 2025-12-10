# src/QISO_CUDA_Kernels.py

import cupy as cp
from numba import cuda
import numpy as np
import math

THREADS_PER_BLOCK = 256

@cuda.jit
def static_collision_kernel(pos, obstacles, N_uavs, N_waypoints, obstacle_dim, penalty_base, result_f2):
    """ Tính toán chi phí va chạm tĩnh. """
    
    idx = cuda.grid(1) 
    if idx < pos.shape[0]:
        total_penalty = 0.0
        
        for i in range(N_uavs * N_waypoints):
            x_w = pos[idx, i * 3 + 0]
            y_w = pos[idx, i * 3 + 1]
            z_w = pos[idx, i * 3 + 2]
            
            for j in range(obstacle_dim):
                x_o = obstacles[j, 0]
                y_o = obstacles[j, 1]
                z_o = obstacles[j, 2]
                r_o = obstacles[j, 3] # Bán kính an toàn
                
                dist_sq = (x_w - x_o)**2 + (y_w - y_o)**2 + (z_w - z_o)**2
                dist = math.sqrt(dist_sq)

                if dist < r_o:
                    # Hàm phạt: Tăng lên khi vi phạm sâu hơn
                    total_penalty += penalty_base * (r_o - dist)
        
        cuda.atomic.add(result_f2, idx, total_penalty) # Sử dụng atomic add để đảm bảo an toàn

@cuda.jit
def dynamic_collision_kernel(pos, N_uavs, N_waypoints, min_sep_sq, penalty_base, result_f2):
    """
    Tính toán chi phí va chạm động (giữa các UAV).
    Kiểm tra khoảng cách giữa UAV_i và UAV_k tại cùng một thời điểm (waypoint j).
    """
    idx = cuda.grid(1)
    if idx < pos.shape[0]:
        total_dynamic_penalty = 0.0
        min_sep = math.sqrt(min_sep_sq)
        
        # Lặp qua từng thời điểm (waypoint)
        for j in range(N_waypoints):
            # Lặp qua tất cả các cặp UAV (i, k)
            for i in range(N_uavs):
                for k in range(i + 1, N_uavs): # Chỉ kiểm tra i < k
                    
                    # Tính offset cho UAV i và k tại waypoint j
                    offset_i = i * N_waypoints * 3 + j * 3
                    offset_k = k * N_waypoints * 3 + j * 3
                    
                    x_i = pos[idx, offset_i + 0]
                    y_i = pos[idx, offset_i + 1]
                    z_i = pos[idx, offset_i + 2]
                    
                    x_k = pos[idx, offset_k + 0]
                    y_k = pos[idx, offset_k + 1]
                    z_k = pos[idx, offset_k + 2]
                    
                    dist_sq = (x_i - x_k)**2 + (y_i - y_k)**2 + (z_i - z_k)**2
                    
                    if dist_sq < min_sep_sq:
                        dist = math.sqrt(dist_sq)
                        # Hàm phạt va chạm động
                        total_dynamic_penalty += penalty_base * (min_sep - dist)
                        
        cuda.atomic.add(result_f2, idx, total_dynamic_penalty)


def calculate_f1_cupy(reshaped_pos):
    """ Tính toán Khoảng cách (f1: Energy/Time Cost) - CuPy Vectorization """
    diff = reshaped_pos[:, :, 1:, :] - reshaped_pos[:, :, :-1, :]
    dist_sq = cp.sum(diff**2, axis=3)
    distance_cost = cp.sum(cp.sqrt(dist_sq), axis=(1, 2))
    return distance_cost

def calculate_f3_cupy(reshaped_pos, mission_targets):
    """ Tính toán F3 (Nhiệm vụ) - Hiện tại là placeholder """
    N_particles = reshaped_pos.shape[0]
    return cp.zeros(N_particles, dtype=cp.float32)
