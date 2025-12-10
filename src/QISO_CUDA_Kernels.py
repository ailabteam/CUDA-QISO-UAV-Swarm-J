# src/QISO_CUDA_Kernels.py - Nâng cấp cho Journal (Thêm CPU Kernels)

import cupy as cp
from numba import cuda, jit # Thêm import jit
import numpy as np
import math

THREADS_PER_BLOCK = 256

# =========================================================================
# I. GPU KERNELS (Giữ nguyên)
# =========================================================================

@cuda.jit
def static_collision_kernel(pos, obstacles, N_uavs, N_waypoints, obstacle_dim, penalty_base, result_f2):
    """ Tính toán chi phí va chạm tĩnh trên GPU. """
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
                r_o = obstacles[j, 3]

                dist_sq = (x_w - x_o)**2 + (y_w - y_o)**2 + (z_w - z_o)**2
                dist = math.sqrt(dist_sq)

                if dist < r_o:
                    total_penalty += penalty_base * (r_o - dist)

        cuda.atomic.add(result_f2, idx, total_penalty)


@cuda.jit
def dynamic_collision_kernel(pos, N_uavs, N_waypoints, min_sep_sq, penalty_base, result_f2):
    """ Tính toán chi phí va chạm động trên GPU. """
    idx = cuda.grid(1)
    if idx < pos.shape[0]:
        total_dynamic_penalty = 0.0
        min_sep = math.sqrt(min_sep_sq)

        for j in range(N_waypoints):
            for i in range(N_uavs):
                for k in range(i + 1, N_uavs):

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
                        total_dynamic_penalty += penalty_base * (min_sep - dist)

        cuda.atomic.add(result_f2, idx, total_dynamic_penalty)


# =========================================================================
# II. CPU KERNELS (Mới - Dùng Numba JIT cho so sánh HPC)
# =========================================================================

@jit(nopython=True)
def static_collision_kernel_cpu(pos, obstacles, N_uavs, N_waypoints, obstacle_dim, penalty_base, result_f2):
    """ Tính toán chi phí va chạm tĩnh trên CPU. """
    
    N_particles = pos.shape[0]
    
    for idx in range(N_particles): # Lặp tuần tự qua các hạt
        total_penalty = 0.0

        for i in range(N_uavs * N_waypoints):
            x_w = pos[idx, i * 3 + 0]
            y_w = pos[idx, i * 3 + 1]
            z_w = pos[idx, i * 3 + 2]

            for j in range(obstacle_dim):
                x_o = obstacles[j, 0]
                y_o = obstacles[j, 1]
                z_o = obstacles[j, 2]
                r_o = obstacles[j, 3]

                dist_sq = (x_w - x_o)**2 + (y_w - y_o)**2 + (z_w - z_o)**2
                dist = math.sqrt(dist_sq)

                if dist < r_o:
                    total_penalty += penalty_base * (r_o - dist)

        result_f2[idx] += total_penalty

@jit(nopython=True)
def dynamic_collision_kernel_cpu(pos, N_uavs, N_waypoints, min_sep_sq, penalty_base, result_f2):
    """ Tính toán chi phí va chạm động trên CPU. """
    
    N_particles = pos.shape[0]
    min_sep = math.sqrt(min_sep_sq)
    
    for idx in range(N_particles): # Lặp tuần tự qua các hạt
        total_dynamic_penalty = 0.0

        for j in range(N_waypoints):
            for i in range(N_uavs):
                for k in range(i + 1, N_uavs):

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
                        total_dynamic_penalty += penalty_base * (min_sep - dist)
        
        result_f2[idx] += total_dynamic_penalty


# =========================================================================
# III. CUPY VECTORIZED UTILITIES (Giữ nguyên)
# =========================================================================

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
