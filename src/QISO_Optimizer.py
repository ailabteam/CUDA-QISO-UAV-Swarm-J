# src/QISO_Optimizer.py - Phiên bản Journal (Đã sửa lỗi logic core)

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

        # Phân loại thuật toán: SPSO, L-DPSO (Linear Dynamic), C-DPSO (Chaos Dynamic)
        self.algo_type = params.get('algo_type', 'SPSO')

        # Tham số cho Dynamic/Chaos
        self.W_min, self.W_max = 0.4, 0.9
        self.C_min, self.C_max = 0.5, 2.5
        self.mu = 4.0 # Chaotic control parameter (Logistic Map)

        # Khởi tạo trạng thái ban đầu của Chaos Map
        self.chaos_state = 0.707106781

        # Khởi tạo toàn bộ quần thể (Vị trí, Vận tốc, Pbest, Gbest)
        self._initialize_particles()
        
        # Thẩm định fitness ban đầu
        self.initial_fitness_check()

    def _initialize_particles(self):
        N = self.env.N_uavs
        M = self.env.N_waypoints
        D = N * M * 3 # Kích thước không gian tìm kiếm

        # 1. Khởi tạo Vị trí (X)
        self.X = self.cp.random.uniform(
            self.env.params['min_bound'],
            self.env.params['max_bound'],
            size=(self.N_particles, D),
            dtype=self.cp.float32
        )

        # Ứng dụng ràng buộc vị trí bắt đầu cố định
        start_pos_np = np.array(self.env.params['start_pos']).flatten()
        start_pos_cp = self.cp.asarray(start_pos_np, dtype=self.cp.float32)

        for i in range(N):
            start_idx = i * M * 3
            # Gán 3 chiều đầu tiên (x, y, z) cho mỗi UAV
            self.X[:, start_idx:start_idx+3] = start_pos_cp[i*3 : i*3+3]

        # 2. Khởi tạo Vận tốc (V)
        self.V = self.cp.random.uniform(-0.5, 0.5, size=(self.N_particles, D), dtype=self.cp.float32)

        # 3. Khởi tạo P_best và G_best (Sẽ được cập nhật sau)
        self.P_best_pos = self.X.copy()
        self.P_best_fitness = self.cp.full(self.N_particles, self.cp.inf, dtype=self.cp.float32)

        # G_best (Được cập nhật từ P_best sau khi đánh giá fitness lần đầu)
        self.G_best_position = self.X[0, :].copy()
        self.G_best_fitness = self.cp.array([self.cp.inf], dtype=self.cp.float32)
        
        self.convergence_history = []
        
    def initial_fitness_check(self):
        """ Tính toán fitness ban đầu và thiết lập Pbest/Gbest lần đầu. """
        
        # Đánh giá fitness ban đầu (trên GPU)
        initial_fitness_np = self.env.evaluate_fitness(self.X) # Trả về NumPy array
        initial_fitness = self.cp.asarray(initial_fitness_np) # Chuyển lại về CuPy

        self.P_best_fitness = initial_fitness.copy()
        
        # Cập nhật G_best từ P_best tốt nhất
        min_idx = self.cp.argmin(initial_fitness)
        self.G_best_fitness[0] = initial_fitness[min_idx]
        self.G_best_position = self.X[min_idx, :].copy()
        
        self.convergence_history.append(float(self.G_best_fitness.get()))


    def _update_dynamic_parameters(self, current_iter):
        t = current_iter
        T_max = self.max_iter
        
        # (Giữ nguyên logic SPSO/L-DPSO/C-DPSO đã được sửa)
        
        # 1. Tính toán Chaotic Value (x_t)
        if self.algo_type == 'C-DPSO':
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

        return W, C1, C2

    def _update_pbest_gbest(self, current_fitness):
        """ Cập nhật Pbest và Gbest dựa trên fitness hiện tại. """
        
        # 1. Cập nhật Pbest
        improved_mask = current_fitness < self.P_best_fitness
        
        # Cập nhật vị trí Pbest
        self.P_best_pos[improved_mask] = self.X[improved_mask].copy()
        
        # Cập nhật fitness Pbest
        self.P_best_fitness[improved_mask] = current_fitness[improved_mask]
        
        # 2. Cập nhật Gbest
        min_pbest_fitness = self.P_best_fitness.min()
        
        if min_pbest_fitness < self.G_best_fitness[0]:
            min_idx = self.cp.argmin(self.P_best_fitness)
            self.G_best_fitness[0] = min_pbest_fitness
            self.G_best_position = self.P_best_pos[min_idx, :].copy()


    def _update_velocity_position(self, W, C1, C2):
        """ Cập nhật vận tốc và vị trí cho toàn bộ quần thể (vectorized). """
        
        r1 = self.cp.random.rand(self.N_particles, self.X.shape[1], dtype=self.cp.float32)
        r2 = self.cp.random.rand(self.N_particles, self.X.shape[1], dtype=self.cp.float32)
        
        # Mở rộng Gbest cho toàn bộ quần thể để tính toán
        G_best_expanded = self.cp.tile(self.G_best_position, (self.N_particles, 1))
        
        # Cập nhật Vận tốc (Equation 5)
        self.V = (W * self.V) + \
                 (C1 * r1 * (self.P_best_pos - self.X)) + \
                 (C2 * r2 * (G_best_expanded - self.X))
        
        # Áp dụng giới hạn Vận tốc (nếu có, không áp dụng trong code này)
        
        # Cập nhật Vị trí (Equation 6)
        self.X = self.X + self.V
        
        # Áp dụng ràng buộc vị trí (boundary constraints)
        X_min = self.env.params['min_bound']
        X_max = self.env.params['max_bound']
        self.X = self.cp.clip(self.X, X_min, X_max)
        
        # Tái áp dụng ràng buộc vị trí bắt đầu cố định (Rất quan trọng!)
        N = self.env.N_uavs
        M = self.env.N_waypoints
        start_pos_np = np.array(self.env.params['start_pos']).flatten()
        start_pos_cp = self.cp.asarray(start_pos_np, dtype=self.cp.float32)
        
        for i in range(N):
            start_idx = i * M * 3
            self.X[:, start_idx:start_idx+3] = start_pos_cp[i*3 : i*3+3]

    def optimize(self):
        
        for t in range(1, self.max_iter + 1):
            
            # 1. Cập nhật tham số động/cố định
            W, C1, C2 = self._update_dynamic_parameters(t) 
            
            # 2. Cập nhật vận tốc và vị trí
            self._update_velocity_position(W, C1, C2)
            
            # 3. Đánh giá fitness
            current_fitness_np = self.env.evaluate_fitness(self.X)
            current_fitness = self.cp.asarray(current_fitness_np)
            
            # 4. Cập nhật Pbest và Gbest
            self._update_pbest_gbest(current_fitness)
            
            self.convergence_history.append(float(self.G_best_fitness.get()))

        return self.G_best_position, self.G_best_fitness, self.convergence_history
