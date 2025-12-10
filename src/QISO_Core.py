# src/QISO_Core.py

import cupy as cp
import numpy as np

class QISO_Optimizer:

    def __init__(self, env, N_particles, max_iter, params, cp):
        self.env = env
        self.cp = cp
        self.N_particles = N_particles
        self.max_iter = max_iter
        self.dim = env.N_uavs * env.N_waypoints * 3

        # --- Tham số thuật toán ---
        # Tham số SPSO Cố định (được dùng nếu is_qiso=False)
        self.C1_fixed = params.get('C1', 1.5)
        self.C2_fixed = params.get('C2', 1.5)
        self.W_fixed = params.get('W', 0.7)

        self.is_qiso = params.get('is_qiso', True)

        # Tham số Chaos (cho QISO dynamic adjustment)
        self.chaotic_map_param = params.get('chaos_mu', 4.0)
        self.chaos_state = cp.array([0.5], dtype=cp.float32)

        # Tham số Dynamic W, C1, C2 (sẽ được cập nhật mỗi iteration nếu is_qiso=True)
        self.W = self.W_fixed
        self.C1 = self.C1_fixed
        self.C2 = self.C2_fixed

        # Ràng buộc không gian
        self.min_b = env.params['min_bound']
        self.max_b = env.params['max_bound']

        # Khởi tạo Vị trí ban đầu
        self.positions = self.cp.random.uniform(
            low=self.min_b,
            high=self.max_b,
            size=(N_particles, self.dim)
        ).astype(self.cp.float32)

        # Khởi tạo Vận tốc nếu là SPSO/QISO_Dynamic
        self.velocities = self.cp.zeros((N_particles, self.dim), dtype=self.cp.float32)

        # Gán Waypoint 1 (Start Position) cố định
        self._apply_start_position_constraint()

        # P_best/G_best và Log
        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = self.cp.full(N_particles, self.cp.inf)
        self.gbest_position = None
        self.gbest_fitness = self.cp.inf
        self.gbest_history = [] # Log hội tụ

    def _apply_start_position_constraint(self):
        # Đảm bảo Waypoint 1 luôn cố định (chạy trong mỗi lần lặp hoặc khởi tạo)
        for i in range(self.env.N_uavs):
             start_coords = self.cp.array(self.env.params['start_pos'][i], dtype=self.cp.float32)
             self.positions[:, i*self.env.N_waypoints*3 : i*self.env.N_waypoints*3 + 3] = start_coords

    def optimize(self):
        algo_type = "QISO" if self.is_qiso else "SPSO"

        for t in range(self.max_iter):

            if self.is_qiso:
                # 1. Tính toán trạng thái Hỗn loạn MỚI (Logistic Map)
                self.chaos_state = self.chaotic_map_param * self.chaos_state * (1.0 - self.chaos_state)
                # 2. Điều chỉnh W, C1, C2 động
                self._dynamic_parameter_adjustment(t)

            current_fitness = self.env.evaluate_fitness(self.positions)
            cp_current_fitness = self.cp.array(current_fitness)

            # Cập nhật P_best
            better_indices = cp_current_fitness < self.pbest_fitness
            self.pbest_fitness = self.cp.where(better_indices, cp_current_fitness, self.pbest_fitness)
            self.pbest_positions[better_indices] = self.positions[better_indices]

            # Cập nhật G_best
            min_idx = self.cp.argmin(cp_current_fitness)
            if cp_current_fitness[min_idx] < self.gbest_fitness:
                self.gbest_fitness = cp_current_fitness[min_idx]
                self.gbest_position = self.positions[min_idx].copy()

            # GHI LOG HỘI TỤ
            self.gbest_history.append(self.gbest_fitness.item())

            # Cập nhật vị trí
            self._update()

            if t % 50 == 0:
                print(f"[{algo_type}] Iteration {t}/{self.max_iter} - G_Best Fitness: {self.gbest_fitness.item():.2f}")

        return self.gbest_position.get(), self.gbest_fitness.item(), self.gbest_history

    def _dynamic_parameter_adjustment(self, t):
        """ Điều chỉnh W, C1, C2 động dựa trên Chaos và Iteration (chỉ cho QISO). """

        chaos_val = self.chaos_state.item() # Lấy giá trị float từ GPU
        
        W_min = 0.4
        W_max = 0.9

                # 1. Quán tính W
        decay = W_max - (W_max - W_min) * (t / self.max_iter)
        W_new = decay * (0.5 + 0.5 * chaos_val)
        
        C_max = 2.5
        C_min = 0.5
        
        # 2. C1 (Cognitive)
        C1_new = C_min + (C_max - C_min) * (1 - (t / self.max_iter)) * chaos_val 

        # 3. C2 (Social)
        C2_new = C_min + (C_max - C_min) * (t / self.max_iter) * chaos_val 
        
        # Áp dụng giá trị mới, sử dụng NumPy clip cho float:
        # NOTE: self.W, self.C1, self.C2 phải được lưu trữ dưới dạng float Python 
        # để tránh xung đột với CuPy, và chỉ được sử dụng dưới dạng float trong _update().

        self.W = np.clip(W_new, W_min, W_max)
        self.C1 = np.clip(C1_new, C_min, C_max)
        self.C2 = np.clip(C2_new, C_min, C_max)



    def _update(self):
        """ Thực hiện công thức cập nhật vị trí và vận tốc SPSO cổ điển. """

        # Lấy tham số W, C1, C2 tương ứng (đã được tính toán là float Python)
        if self.is_qiso:
            W = self.W
            C1 = self.C1
            C2 = self.C2
        else:
            W = self.W_fixed
            C1 = self.C1_fixed
            C2 = self.C2_fixed

        # Khởi tạo r1, r2 dựa trên C1, C2 hiện tại (CuPy tự động nhận float Python)
        r1 = self.cp.random.uniform(0, C1, size=(self.N_particles, 1))
        r2 = self.cp.random.uniform(0, C2, size=(self.N_particles, 1))

        gbest_expanded = self.cp.tile(self.gbest_position, (self.N_particles, 1))

        # Cập nhật Vận tốc
        cognitive = r1 * (self.pbest_positions - self.positions)
        social = r2 * (gbest_expanded - self.positions)

        self.velocities = W * self.velocities + cognitive + social

        # Cập nhật Vị trí
        self.positions += self.velocities

        # Áp dụng ràng buộc không gian và ràng buộc cứng
        self.positions = self.cp.clip(self.positions, self.min_b, self.max_b)
        self._apply_start_position_constraint()
