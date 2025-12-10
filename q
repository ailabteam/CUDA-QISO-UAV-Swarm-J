(qiso_uav) daipv11@Ubuntu22:~/CUDA-QISO-UAV-Swarm-J$ cat src/QISO_Optimizer.py
# src/QISO_Core.py - Nâng cấp cho Journal (Thêm L-DPSO)

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

        # Khởi tạo trạng thái ban đầu của Chaos Map (cho C-DPSO)
        # Bắt đầu với một giá trị ngẫu nhiên không phải 0.0, 0.25, 0.5, 0.75, 1.0
        self.chaos_state = 0.707106781 # sqrt(0.5) ~ 0.707

        # ... (Các khởi tạo khác)
        self._initialize_particles()

    def _initialize_particles(self):
        N = self.env.N_uavs
        M = self.env.N_waypoints
        D = N * M * 3 # Kích thước không gian tìm kiếm

        # 1. Khởi tạo Vị trí (X)
        # Random Uniform trong phạm vi Max/Min bounds
        self.X = self.cp.random.uniform(
            self.env.params['min_bound'], 
            self.env.params['max_bound'], 
            size=(self.N_particles, D), 
            dtype=self.cp.float32
        )
        
        # Áp dụng ràng buộc vị trí bắt đầu (W_i,1)
        start_pos_np = np.array(self.env.params['start_pos']).flatten()
        # Chuyển về CuPy
        start_pos_cp = self.cp.asarray(start_pos_np, dtype=self.cp.float32)

        # Lặp qua tất cả hạt và gán vị trí bắt đầu cố định (3 chiều đầu tiên của mỗi UAV)
        for i in range(N):
            start_idx = i * M * 3
            self.X[:, start_idx:start_idx+3] = start_pos_cp[i*3 : i*3+3]

        # 2. Khởi tạo Vận tốc (V)
        self.V = self.cp.random.uniform(-0.5, 0.5, size=(self.N_particles, D), dtype=self.cp.float32)
        
        # 3. Khởi tạo P_best và G_best
        self.P_best_pos = self.X.copy()
        self.P_best_fitness = self.cp.full(self.N_particles, self.cp.inf, dtype=self.cp.float32)
        
        # G_best chỉ cần một vị trí và một giá trị fitness
        self.G_best_position = self.X[0, :].copy()
        self.G_best_fitness = self.cp.array([self.cp.inf], dtype=self.cp.float32)

        self.convergence_history = []


    def _update_dynamic_parameters(self, current_iter):
        t = current_iter
        T_max = self.max_iter

        # 1. Tính toán Chaotic Value (x_t)
        if self.algo_type == 'C-DPSO':
            # Cập nhật Logistic Chaos Map
            self.chaos_state = self.mu * self.chaos_state * (1 - self.chaos_state)
            x_t = self.chaos_state
        else:
            x_t = 1.0 # Giá trị trung lập/không tác động nếu không dùng Chaos

        # 2. Tính toán W (Inertia Weight)
        W_linear = self.W_max - (self.W_max - self.W_min) * (t / T_max)

        if self.algo_type == 'SPSO':
            W = self.params['W']
        elif self.algo_type == 'L-DPSO':
            W = W_linear # Pure Dynamic Linear
        elif self.algo_type == 'C-DPSO':
            # Chaos-Modulated Dynamic
            W = W_linear * (0.5 + 0.5 * x_t)
        else:
            W = self.params['W']

        # 3. Tính toán C1 (Cognitive) và C2 (Social)
        C_scale = self.C_max - self.C_min

        if self.algo_type == 'SPSO':
            C1 = self.params['C1']
            C2 = self.params['C2']
        elif self.algo_type == 'L-DPSO':
            # Pure Dynamic Linear
            C1 = self.C_min + C_scale * (1 - t / T_max)
            C2 = self.C_min + C_scale * (t / T_max)
        elif self.algo_type == 'C-DPSO':
            # Chaos-Modulated Dynamic
            C1 = self.C_min + C_scale * (1 - t / T_max) * x_t
            C2 = self.C_min + C_scale * (t / T_max) * x_t
        else:
            C1 = self.params['C1']
            C2 = self.params['C2']

        return W, C1, C2

    def _update_velocity_position(self, W, C1, C2):
        # ... (logic giữ nguyên)
        pass # Giả định logic cập nhật vị trí/vận tốc đã có

    def optimize(self):
        # Khởi tạo các biến ... (giữ nguyên)

        for t in range(1, self.max_iter + 1):

            # 1. Cập nhật tham số động/cố định
            W, C1, C2 = self._update_dynamic_parameters(t)

            # 2. Cập nhật vận tốc và vị trí
            self._update_velocity_position(W, C1, C2)

            # 3. Đánh giá fitness
            current_fitness = self.env.evaluate_fitness(self.X)

            # 4. Cập nhật Pbest và Gbest
            # ... (logic giữ nguyên)

            self.convergence_history.append(float(self.G_best_fitness.get()))

        return self.G_best_position, self.G_best_fitness, self.convergence_history
