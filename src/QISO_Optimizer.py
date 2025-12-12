# src/QISO_Optimizer.py - Phiên bản Journal HOÀN CHỈNH

import cupy as cp
import numpy as np
import math
import time
from src.Environment import UAV_Environment

class QISO_Optimizer:

    def __init__(self, env, N_particles, max_iter, params, cp):

        self.env = env
        self.cp = cp
        self.N_particles = N_particles
        self.max_iter = max_iter
        self.params = params

        self.algo_type = params.get('algo_type', 'SPSO')

        self.W_min, self.W_max = 0.4, 0.9
        self.C_min, self.C_max = 0.5, 2.5
        self.mu = 4.0

        self.chaos_state = 0.707106781

        # Khởi tạo các list lưu trữ lịch sử
        self.W_history = []
        self.C1_history = []
        self.C2_history = []

        self._initialize_particles()
        self.initial_fitness_check()

    def _initialize_particles(self):
        N = self.env.N_uavs
        M = self.env.N_waypoints
        D = N * M * 3

        # LẤY GIỚI HẠN D-DIMENSIONAL TỪ ENVIRONMENT
        min_b = self.env.params['min_bound_D']
        max_b = self.env.params['max_bound_D']
        
        random_base = self.cp.random.rand(self.N_particles, D, dtype=self.cp.float32)
        
        # Mở rộng giới hạn (để match kích thước N_particles x D)
        min_b_expanded = self.cp.tile(min_b, (self.N_particles, 1))
        max_b_expanded = self.cp.tile(max_b, (self.N_particles, 1))

        # Khởi tạo X: X = min_b + rand * (max_b - min_b)
        self.X = min_b_expanded + random_base * (max_b_expanded - min_b_expanded)

        # Cố định điểm bắt đầu (logic giữ nguyên)
        start_pos_np = np.array(self.env.params['start_pos']).flatten()
        start_pos_cp = self.cp.asarray(start_pos_np, dtype=self.cp.float32)

        for i in range(N):
            start_idx = i * M * 3
            self.X[:, start_idx:start_idx+3] = start_pos_cp[i*3 : i*3+3]

        self.V = self.cp.random.uniform(-0.5, 0.5, size=(self.N_particles, D), dtype=self.cp.float32)

        self.P_best_pos = self.X.copy()
        self.P_best_fitness = self.cp.full(self.N_particles, self.cp.inf, dtype=self.cp.float32)

        self.G_best_position = self.X[0, :].copy()
        self.G_best_fitness = self.cp.array([self.cp.inf], dtype=self.cp.float32)

        self.convergence_history = []


    def initial_fitness_check(self):
        initial_fitness_np = self.env.evaluate_fitness(self.X)
        initial_fitness = self.cp.asarray(initial_fitness_np)

        self.P_best_fitness = initial_fitness.copy()

        min_idx = self.cp.argmin(initial_fitness)
        self.G_best_fitness[0] = initial_fitness[min_idx]
        self.G_best_position = self.X[min_idx, :].copy()

        self.convergence_history.append(float(self.G_best_fitness.get()))

        # Ghi lại tham số ban đầu (t=0)
        W, C1, C2 = self._update_dynamic_parameters(0)


    def _update_dynamic_parameters(self, current_iter):
        t = current_iter
        T_max = self.max_iter

        # 1. Tính toán Chaotic Value (x_t)
        if self.algo_type == 'C-DPSO' and t > 0:
            self.chaos_state = self.mu * self.chaos_state * (1 - self.chaos_state)
            x_t = self.chaos_state
        else:
            x_t = 1.0

        # 2. Tính toán W (Inertia Weight)
        W_linear = self.W_max - (self.W_max - self.W_min) * (t / T_max)

        if self.algo_type == 'SPSO':
            W = self.params['W']
        elif self.algo_type == 'L-DPSO':
            W = W_linear
        elif self.algo_type == 'C-DPSO':
            W = W_linear * (0.5 + 0.5 * x_t)
        else:
            W = self.params['W']

        # 3. Tính toán C1 (Cognitive) và C2 (Social)
        C_scale = self.C_max - self.C_min

        if self.algo_type == 'SPSO':
            C1 = self.params['C1']
            C2 = self.params['C2']
        elif self.algo_type == 'L-DPSO':
            C1 = self.C_min + C_scale * (1 - t / T_max)
            C2 = self.C_min + C_scale * (t / T_max)
        elif self.algo_type == 'C-DPSO':
            C1 = self.C_min + C_scale * (1 - t / T_max) * x_t
            C2 = self.C_min + C_scale * (t / T_max) * x_t
        else:
            C1 = self.params['C1']
            C2 = self.params['C2']

        # Lưu lịch sử tham số (cho Figure 4)
        self.W_history.append(float(W))
        self.C1_history.append(float(C1))
        self.C2_history.append(float(C2))

        return W, C1, C2

    def _update_pbest_gbest(self, current_fitness):
        # ... (logic giữ nguyên)

        improved_mask = current_fitness < self.P_best_fitness
        self.P_best_pos[improved_mask] = self.X[improved_mask].copy()
        self.P_best_fitness[improved_mask] = current_fitness[improved_mask]

        min_pbest_fitness = self.P_best_fitness.min()

        if min_pbest_fitness < self.G_best_fitness[0]:
            min_idx = self.cp.argmin(self.P_best_fitness)
            self.G_best_fitness[0] = min_pbest_fitness
            self.G_best_position = self.P_best_pos[min_idx, :].copy()

    def _update_velocity_position(self, W, C1, C2):
        
        r1 = self.cp.random.rand(self.N_particles, self.X.shape[1], dtype=self.cp.float32)
        r2 = self.cp.random.rand(self.N_particles, self.X.shape[1], dtype=self.cp.float32)
        G_best_expanded = self.cp.tile(self.G_best_position, (self.N_particles, 1))

        self.V = (W * self.V) + \
                 (C1 * r1 * (self.P_best_pos - self.X)) + \
                 (C2 * r2 * (G_best_expanded - self.X))

        self.X = self.X + self.V

        # ÁP DỤNG CLIPPING BẰNG GIỚI HẠN D-DIMENSIONAL MỚI
        X_min = self.env.params['min_bound_D']
        X_max = self.env.params['max_bound_D']
        self.X = self.cp.clip(self.X, X_min, X_max)

        # Cố định điểm bắt đầu (logic giữ nguyên)
        N = self.env.N_uavs
        M = self.env.N_waypoints
        start_pos_np = np.array(self.env.params['start_pos']).flatten()
        start_pos_cp = self.cp.asarray(start_pos_np, dtype=self.cp.float32)

        for i in range(N):
            start_idx = i * M * 3
            self.X[:, start_idx:start_idx+3] = start_pos_cp[i*3 : i*3+3]
            
            
    def optimize(self):

        for t in range(1, self.max_iter + 1):

            W, C1, C2 = self._update_dynamic_parameters(t)
            self._update_velocity_position(W, C1, C2)

            current_fitness_np = self.env.evaluate_fitness(self.X)
            current_fitness = self.cp.asarray(current_fitness_np)

            self._update_pbest_gbest(current_fitness)

            self.convergence_history.append(float(self.G_best_fitness.get()))

        # === ÁP DỤNG RÀNG BUỘC CUỐI CÙNG TRÊN NGHIỆM TỐT NHẤT ===
        X_min = self.env.params['min_bound_D']
        X_max = self.env.params['max_bound_D']

        # Đảm bảo nghiệm tốt nhất không vượt biên do sai số dấu phẩy động
        self.G_best_position = self.cp.clip(self.G_best_position, X_min, X_max)

        # Trả về G_best_position đã được clip
        return self.G_best_position, self.G_best_fitness, self.convergence_history, self.W_history, self.C1_history, self.C2_history
    

        for t in range(1, self.max_iter + 1):

            W, C1, C2 = self._update_dynamic_parameters(t)
            self._update_velocity_position(W, C1, C2)

            current_fitness_np = self.env.evaluate_fitness(self.X)
            current_fitness = self.cp.asarray(current_fitness_np)

            self._update_pbest_gbest(current_fitness)

            self.convergence_history.append(float(self.G_best_fitness.get()))

        # === BỔ SUNG: ÁP DỤNG RÀNG BUỘC CUỐI CÙNG TRÊN NGHIỆM TỐT NHẤT ===
        X_min = self.env.params['min_bound']
        X_max = self.env.params['max_bound']

        # Đảm bảo nghiệm tốt nhất không vượt biên do sai số dấu phẩy động
        self.G_best_position = self.cp.clip(self.G_best_position, X_min, X_max)

        # Trả về G_best_position đã được clip
        return self.G_best_position, self.G_best_fitness, self.convergence_history, self.W_history, self.C1_history, self.C2_history
