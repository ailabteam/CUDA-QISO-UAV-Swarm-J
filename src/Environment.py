# src/Environment.py - Phiên bản Journal (Hoàn chỉnh)

import cupy as cp
import numpy as np
from src.QISO_CUDA_Kernels import static_collision_kernel, dynamic_collision_kernel, calculate_f1_cupy, calculate_f3_cupy, THREADS_PER_BLOCK
from src.QISO_CUDA_Kernels import static_collision_kernel_cpu, dynamic_collision_kernel_cpu
import math

# Hình phạt biên cứng (Hard Penalty)
BOUNDARY_WEIGHT = 500000.0

# Hàm tính penalty biên (Sử dụng CuPy)
def calculate_boundary_penalty_cupy(positions, min_bound, max_bound):
    """
    Tính toán hình phạt biên cứng (Hard Penalty) cho GPU.
    Nếu bất kỳ chiều nào vượt biên, phạt lớn gấp bội số lần vi phạm.
    """
    # Tìm các vi phạm (giá trị > 0 nếu vi phạm)
    violations_min = cp.maximum(min_bound - positions, 0)
    violations_max = cp.maximum(positions - max_bound, 0)

    # Tổng hợp độ lớn vi phạm trên mỗi hạt (particle)
    total_violation_sum = cp.sum(violations_min + violations_max, axis=1)

    # Áp dụng trọng số penalty lớn
    boundary_penalty = BOUNDARY_WEIGHT * total_violation_sum

    return boundary_penalty

# Hàm tính penalty biên (Sử dụng NumPy)
def calculate_boundary_penalty_numpy(positions_np, min_bound, max_bound):
    """ Tính toán hình phạt biên cứng cho CPU. """
    violations_min = np.maximum(min_bound - positions_np, 0)
    violations_max = np.maximum(positions_np - max_bound, 0)
    total_violation_sum = np.sum(violations_min + violations_max, axis=1)

    boundary_penalty = BOUNDARY_WEIGHT * total_violation_sum
    return boundary_penalty


class UAV_Environment:

    def __init__(self, N_uavs, N_waypoints, config, cp):
        self.cp = cp
        self.N_uavs = N_uavs
        self.N_waypoints = N_waypoints

        self.config = config
        self.params = config['sim_params']
        
        # Gọi hàm tính toán giới hạn chính xác cho PSO
        self._calculate_d_bounds()

        self.obstacles = self.cp.array(config['obstacles_data'], dtype=self.cp.float32)
        self.obstacles_np = np.array(config['obstacles_data'], dtype=np.float32)

        self.mission_targets = self.cp.array(config['mission_targets'], dtype=self.cp.float32)
        self.mission_targets_np = np.array(config['mission_targets'], dtype=np.float32)

        self.min_separation = self.params.get('min_separation', 0.0)

        N_particles = config['qiso_params']['N_particles']
        self.blocks = (N_particles + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    
    def _calculate_d_bounds(self):
        """ Tính toán mảng giới hạn 1D (D-dimensional) cho PSO (360 chiều). """
        
        # Giả định: dimensions = [X_max, Y_max, Z_max] (ví dụ: [1000, 1000, 500])
        bounds = self.params['dimensions'] 
        X_max, Y_max, Z_max = bounds[0], bounds[1], bounds[2]
        X_min, Y_min, Z_min = 0.0, 0.0, 0.0 
        
        # Mảng 1D (3 chiều) chứa giới hạn cho một waypoint: [X, Y, Z]
        min_dim_xyz = np.array([X_min, Y_min, Z_min], dtype=np.float32)
        max_dim_xyz = np.array([X_max, Y_max, Z_max], dtype=np.float32)

        # Mở rộng thành mảng D-dimensional (D=360)
        min_bound_template = np.tile(min_dim_xyz, self.N_waypoints * self.N_uavs)
        max_bound_template = np.tile(max_dim_xyz, self.N_waypoints * self.N_uavs)

        # Lưu vào params để QISO_Optimizer có thể truy cập
        self.params['min_bound_D'] = self.cp.asarray(min_bound_template)
        self.params['max_bound_D'] = self.cp.asarray(max_bound_template)
    
    def evaluate_fitness(self, positions):
        """ Đánh giá Fitness sử dụng GPU (CuPy/CUDA) """
        N_particles = positions.shape[0]
        reshaped_pos = positions.reshape(N_particles, self.N_uavs, self.N_waypoints, 3)

        # 1. F1 (Khoảng cách)
        distance_cost = calculate_f1_cupy(reshaped_pos)

        # 2. F2 (Va chạm Tĩnh & Động)
        collision_penalty_f2 = self.cp.zeros(N_particles, dtype=self.cp.float32)
        penalty_base_value = self.params['weights'][1]

        # [Kernel calls for static and dynamic collision...]
        obstacle_count = self.obstacles.shape[0]
        if obstacle_count > 0:
            static_collision_kernel[self.blocks, THREADS_PER_BLOCK](
                positions, self.obstacles, self.N_uavs, self.N_waypoints, obstacle_count,
                penalty_base_value * 50, collision_penalty_f2
            )
        if self.min_separation > 0.0:
            min_sep_sq = self.min_separation**2
            dynamic_collision_kernel[self.blocks, THREADS_PER_BLOCK](
                positions, self.N_uavs, self.N_waypoints, min_sep_sq,
                penalty_base_value * 100, collision_penalty_f2
            )

        # 3. F3 (Nhiệm vụ)
        task_penalty = self.cp.zeros(N_particles, dtype=self.cp.float32)

        # 4. F4 (Ràng buộc Biên - SỬ DỤNG MẢNG D-DIMENSIONAL)
        min_b = self.params['min_bound_D']
        max_b = self.params['max_bound_D']
        f_boundary = calculate_boundary_penalty_cupy(positions, min_b, max_b)

        # Áp dụng trọng số
        w1, _, w3 = self.params['weights']
        fitness = w1 * distance_cost + collision_penalty_f2 + w3 * task_penalty + f_boundary

        return fitness.get()
    
        """ Đánh giá Fitness sử dụng GPU (CuPy/CUDA) """
        N_particles = positions.shape[0]
        reshaped_pos = positions.reshape(N_particles, self.N_uavs, self.N_waypoints, 3)

        # 1. F1 (Khoảng cách)
        distance_cost = calculate_f1_cupy(reshaped_pos)

        # 2. F2 (Va chạm Tĩnh & Động)
        collision_penalty_f2 = self.cp.zeros(N_particles, dtype=self.cp.float32)
        penalty_base_value = self.params['weights'][1]

        # [Kernel calls for static and dynamic collision...]
        obstacle_count = self.obstacles.shape[0]
        if obstacle_count > 0:
            static_collision_kernel[self.blocks, THREADS_PER_BLOCK](
                positions, self.obstacles, self.N_uavs, self.N_waypoints, obstacle_count,
                penalty_base_value * 50, collision_penalty_f2
            )
        if self.min_separation > 0.0:
            min_sep_sq = self.min_separation**2
            dynamic_collision_kernel[self.blocks, THREADS_PER_BLOCK](
                positions, self.N_uavs, self.N_waypoints, min_sep_sq,
                penalty_base_value * 100, collision_penalty_f2
            )

        # 3. F3 (Nhiệm vụ)
        task_penalty = self.cp.zeros(N_particles, dtype=self.cp.float32)

        # 4. F4 (Ràng buộc Biên - MỚI)
        min_b = self.params['min_bound']
        max_b = self.params['max_bound']
        f_boundary = calculate_boundary_penalty_cupy(positions, min_b, max_b)

        # Áp dụng trọng số
        w1, _, w3 = self.params['weights']
        fitness = w1 * distance_cost + collision_penalty_f2 + w3 * task_penalty + f_boundary

        return fitness.get()

    def evaluate_fitness_cpu(self, positions_np):
        """ Đánh giá Fitness sử dụng CPU (NumPy/Numba JIT) """

        N_particles = positions_np.shape[0]

        # 1. Tính F1 (Khoảng cách) - Giữ nguyên logic tính toán

        reshaped_pos = positions_np.reshape(N_particles, self.N_uavs, self.N_waypoints, 3)
        diff = reshaped_pos[:, :, 1:, :] - reshaped_pos[:, :, :-1, :]
        dist_sq = np.sum(diff**2, axis=3)
        distance_cost = np.sum(np.sqrt(dist_sq), axis=(1, 2))


        # 2. F2 (Va chạm Tĩnh & Động) - Giữ nguyên logic tính toán
        collision_penalty_f2 = np.zeros(N_particles, dtype=np.float32)
        penalty_base_value = self.params['weights'][1]

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

        # 3. F3 (Nhiệm vụ)
        task_penalty = np.zeros(N_particles, dtype=np.float32)

        # 4. F4 (Ràng buộc Biên - SỬ DỤNG MẢNG D-DIMENSIONAL)
        min_b = self.params['min_bound_D'].get()
        max_b = self.params['max_bound_D'].get()
        f_boundary = calculate_boundary_penalty_numpy(positions_np, min_b, max_b)


        # Áp dụng trọng số
        w1, _, w3 = self.params['weights']
        fitness = w1 * distance_cost + collision_penalty_f2 + w3 * task_penalty + f_boundary

        return fitness


        """ Đánh giá Fitness sử dụng CPU (NumPy/Numba JIT) """

        N_particles = positions_np.shape[0]

        # 1. Tính F1 (Khoảng cách) - Dùng NumPy (logic giữ nguyên)
        reshaped_pos = positions_np.reshape(N_particles, self.N_uavs, self.N_waypoints, 3)
        diff = reshaped_pos[:, :, 1:, :] - reshaped_pos[:, :, :-1, :]
        dist_sq = np.sum(diff**2, axis=3)
        distance_cost = np.sum(np.sqrt(dist_sq), axis=(1, 2))

        # 2. F2 (Va chạm Tĩnh & Động) - Dùng NumPy/Numba CPU
        collision_penalty_f2 = np.zeros(N_particles, dtype=np.float32)
        penalty_base_value = self.params['weights'][1]

        # 2a. Va chạm Tĩnh (Static) - Lời gọi HÀM ĐÚNG
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

        # 2b. Va chạm Động (Dynamic - UAV-UAV) - Lời gọi HÀM ĐÚNG
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

        # 3. F3 (Nhiệm vụ)
        task_penalty = np.zeros(N_particles, dtype=np.float32)

        # 4. F4 (Ràng buộc Biên - MỚI)
        min_b = self.params['min_bound']
        max_b = self.params['max_bound']
        f_boundary = calculate_boundary_penalty_numpy(positions_np, min_b, max_b)


        # Áp dụng trọng số
        w1, _, w3 = self.params['weights']
        fitness = w1 * distance_cost + collision_penalty_f2 + w3 * task_penalty + f_boundary

        return fitness



        """ Đánh giá Fitness sử dụng CPU (NumPy/Numba JIT) """

        N_particles = positions_np.shape[0]

        # 1. F1 (Khoảng cách) - Dùng NumPy (logic giữ nguyên)
        reshaped_pos = positions_np.reshape(N_particles, self.N_uavs, self.N_waypoints, 3)
        diff = reshaped_pos[:, :, 1:, :] - reshaped_pos[:, :, :-1, :]
        dist_sq = np.sum(diff**2, axis=3)
        distance_cost = np.sum(np.sqrt(dist_sq), axis=(1, 2))

        # 2. F2 (Va chạm Tĩnh & Động) - Dùng NumPy/Numba CPU (logic giữ nguyên)
        collision_penalty_f2 = np.zeros(N_particles, dtype=np.float32)
        penalty_base_value = self.params['weights'][1]

        if self.obstacles_np.shape[0] > 0:
            static_collision_kernel_cpu(..., collision_penalty_f2) # Giữ nguyên gọi hàm
        if self.min_separation > 0.0:
            min_sep_sq = self.min_separation**2
            dynamic_collision_kernel_cpu(..., collision_penalty_f2) # Giữ nguyên gọi hàm

        # 3. F3 (Nhiệm vụ)
        task_penalty = np.zeros(N_particles, dtype=np.float32)

        # 4. F4 (Ràng buộc Biên - MỚI)
        min_b = self.params['min_bound']
        max_b = self.params['max_bound']
        f_boundary = calculate_boundary_penalty_numpy(positions_np, min_b, max_b)


        # Áp dụng trọng số
        w1, _, w3 = self.params['weights']
        fitness = w1 * distance_cost + collision_penalty_f2 + w3 * task_penalty + f_boundary

        return fitness
