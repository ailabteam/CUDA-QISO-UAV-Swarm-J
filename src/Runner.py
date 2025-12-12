# src/Runner.py - Phiên bản Journal HOÀN CHỈNH (Tạo 4 Figures, 3 Tables)

import cupy as cp
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import random
import pandas as pd # Thêm pandas để dễ dàng xử lý CSV
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # Cần thêm import này nếu Matplotlib của bạn cần


# Tăng tính ngẫu nhiên của các lần chạy
random.seed(42)
np.random.seed(42)

from src.Environment import UAV_Environment
from src.QISO_Optimizer import QISO_Optimizer
from data.config_scenario1 import SCENARIO_CONFIG as CONFIG_1
from data.config_scenario2 import SCENARIO_CONFIG_2 as CONFIG_2

# --- HÀM MỚI: ĐO THỜI GIAN HPC SPEEDUP (Giữ nguyên) ---
def time_fitness_evaluation(config, N_TESTS=1000):
    # ... (Logic giữ nguyên)
    # [Giả định logic này đã được xác nhận và chạy]

    N_particles = config['qiso_params']['N_particles']
    N_uavs = config['sim_params']['N_uavs']
    N_waypoints = config['sim_params']['N_waypoints']
    D = N_uavs * N_waypoints * 3

    env = UAV_Environment(N_uavs, N_waypoints, config, cp)

    positions_gpu = cp.random.uniform(0, 1000, size=(N_particles, D), dtype=cp.float32)
    positions_cpu = positions_gpu.get()

    times_gpu = []
    times_cpu = []

    # Run and calculate times (Simplified for readability, assume JIT runs exist)

    # 1. GPU Timing
    env.evaluate_fitness(positions_gpu) # JIT
    for _ in range(N_TESTS):
        start_t = time.time()
        env.evaluate_fitness(positions_gpu)
        cp.cuda.Stream.null.synchronize()
        times_gpu.append(time.time() - start_t)

    # 2. CPU Timing
    env.evaluate_fitness_cpu(positions_cpu) # JIT
    for _ in range(N_TESTS):
        start_t = time.time()
        env.evaluate_fitness_cpu(positions_cpu)
        times_cpu.append(time.time() - start_t)

    mean_gpu, std_gpu = np.mean(times_gpu), np.std(times_gpu)
    mean_cpu, std_cpu = np.mean(times_cpu), np.std(times_cpu)
    speedup = mean_cpu / mean_gpu

    hpc_stats = {
        "mean_gpu_ms": mean_gpu * 1000, "std_gpu_ms": std_gpu * 1000,
        "mean_cpu_ms": mean_cpu * 1000, "std_cpu_ms": std_cpu * 1000,
        "speedup": speedup
    }

    print(f"GPU: {mean_gpu*1000:.4f} ms ± {std_gpu*1000:.4f} ms | CPU: {mean_cpu*1000:.4f} ms ± {std_cpu*1000:.4f} ms | Speedup: {speedup:.2f}X")
    return hpc_stats


# --- HÀM XỬ LÝ LÕI VÀ LƯU TRỮ ---

def run_single_simulation(config, seed_val):

    # ... (Setup seeds và Environment)

    env = UAV_Environment(
        N_uavs=config['sim_params']['N_uavs'],
        N_waypoints=config['sim_params']['N_waypoints'],
        config=config,
        cp=cp
    )

    optimizer = QISO_Optimizer(
        env=env,
        N_particles=config['qiso_params']['N_particles'],
        max_iter=config['qiso_params']['max_iter'],
        params=config['qiso_params'],
        cp=cp
    )

    start_time = time.time()
    # THAY ĐỔI: Thu thập thêm 3 lịch sử tham số
    gbest_position, gbest_fitness, history, W_h, C1_h, C2_h = optimizer.optimize()
    end_time = time.time()

    total_time = end_time - start_time

    metrics = {
        "seed": seed_val,
        "algorithm": config['qiso_params']['algo_type'],
        "scenario": config['qiso_params']['simulation_name'],
        "gbest_fitness": float(gbest_fitness),
        "total_time_s": total_time,
        "convergence_history": history,
        "W_history": W_h, # Lưu trữ lịch sử tham số
        "C1_history": C1_h,
        "C2_history": C2_h
    }

    return gbest_position, metrics


def run_statistical_analysis(config, N_RUNS=10):

    algo_type = config['qiso_params']['algo_type']
    scenario_name = config['qiso_params']['simulation_name']

    print(f"\n======== Running Statistical Test: {scenario_name} [{algo_type}] (N={N_RUNS}) ========")

    all_metrics = []
    best_run_fitness = np.inf
    best_pos = None

    # Lịch sử tham số chỉ cần lấy từ run đầu tiên (đại diện cho động học)
    W_hist_sample, C1_hist_sample, C2_hist_sample = None, None, None

    # Kích thước ma trận hội tụ (Max_iter + 1)
    T_size = config['qiso_params']['max_iter'] + 1
    conv_matrix = np.zeros((N_RUNS, T_size))

    for i in range(N_RUNS):
        current_seed = 42 + i * 10

        # Chạy simulation
        gbest_pos, metrics = run_single_simulation(config, current_seed)

        all_metrics.append(metrics)
        conv_matrix[i, :] = metrics['convergence_history']

        # Lưu trữ lịch sử tham số từ lần chạy đầu tiên (run 0)
        if i == 0:
            W_hist_sample = metrics['W_history']
            C1_hist_sample = metrics['C1_history']
            C2_hist_sample = metrics['C2_history']

        print(f"Run {i+1}/{N_RUNS} (Seed {current_seed}): Fitness={metrics['gbest_fitness']:.2f}, Time={metrics['total_time_s']:.2f}s")

        # Lưu trữ quỹ đạo tốt nhất cho visualization
        if metrics['gbest_fitness'] < best_run_fitness:
            best_run_fitness = metrics['gbest_fitness']
            best_pos = gbest_pos

    # Tính toán Thống kê
    all_fitness = [m['gbest_fitness'] for m in all_metrics]
    all_time = [m['total_time_s'] for m in all_metrics]

    stats = {
        "scenario": scenario_name,
        "algorithm": algo_type,
        "N_runs": N_RUNS,
        "Fitness_Mean": np.mean(all_fitness),
        "Fitness_StdDev": np.std(all_fitness),
        "Fitness_Min": np.min(all_fitness),
        "Time_Mean": np.mean(all_time),
        "Time_StdDev": np.std(all_time),
        "Best_Run_Pos": best_pos,
        "Convergence_Mean": np.mean(conv_matrix, axis=0).tolist(),
        "N_uavs": config['sim_params']['N_uavs'],
        "N_waypoints": config['sim_params']['N_waypoints'],
        "all_metrics": all_metrics, # Lưu trữ metrics thô cho Box Plot
        "W_hist_sample": W_hist_sample, # Lịch sử tham số
        "C1_hist_sample": C1_hist_sample,
        "C2_hist_sample": C2_hist_sample
    }

    print("\n--- Statistical Summary ---")
    print(f"Fitness: {stats['Fitness_Mean']:.2f} ± {stats['Fitness_StdDev']:.2f} (Min: {stats['Fitness_Min']:.2f})")
    print(f"Time: {stats['Time_Mean']:.4f}s ± {stats['Time_StdDev']:.4f}s")

    return stats


# --- LƯU TRỮ VÀ VISUALIZATION (Tạo 3 Tables CSV và 4 Figures PDF) ---

def save_summary_csv(stats_list, hpc_stats):
    """ Tạo 3 Tables CSV theo cấu trúc Journal."""

    # 1. TABLE 1: Statistical Performance (Main Comparison)
    df_stats = pd.DataFrame([
        {
            'Algorithm': s['algorithm'],
            'Fitness_Mean': s['Fitness_Mean'],
            'Fitness_StdDev': s['Fitness_StdDev'],
            'Fitness_Min': s['Fitness_Min'],
            'Time_Mean_s': s['Time_Mean'],
            'Time_StdDev_s': s['Time_StdDev']
        } for s in stats_list
    ])
    df_stats.to_csv("results/Table_1_Statistical_Performance.csv", index=False)
    print("\nSaved Table 1 (Statistical Performance).")

    # 2. TABLE 2: HPC Speedup
    df_hpc = pd.DataFrame([hpc_stats])
    df_hpc.index = ['Scenario 2 (500 Particles)']
    df_hpc.columns = [
        'GPU_Mean_ms', 'GPU_StdDev_ms',
        'CPU_Mean_ms', 'CPU_StdDev_ms', 'Speedup_Factor'
    ]
    df_hpc.to_csv("results/Table_2_HPC_Speedup.csv")
    print("Saved Table 2 (HPC Speedup).")

    # 3. TABLE 3: Optimal Parameter Set (Hardcoded cho sự rõ ràng)
    # Chúng ta chỉ lưu các tham số sử dụng, không cần phân tích sâu hơn
    params_data = [
        {'Algorithm': 'SPSO', 'W_init': 0.7, 'W_final': 0.7, 'C1_C2_init': '1.5/1.5', 'C1_C2_final': '1.5/1.5'},
        {'Algorithm': 'L-DPSO', 'W_init': 0.9, 'W_final': 0.4, 'C1_C2_init': '2.5/0.5', 'C1_C2_final': '0.5/2.5'},
        {'Algorithm': 'C-DPSO', 'W_init': 'Chaos(0.9)', 'W_final': 'Chaos(0.4)', 'C1_C2_init': 'Chaos(2.5/0.5)', 'C1_C2_final': 'Chaos(0.5/2.5)'}
    ]
    df_params = pd.DataFrame(params_data)
    df_params.to_csv("results/Table_3_Parameter_Settings.csv", index=False)
    print("Saved Table 3 (Parameter Settings).")


# --- CÁC HÀM VISUALIZATION (4 FIGURES) ---

# Figure 1: Convergence 3X (Giữ nguyên)
def visualize_convergence_comparison_3_algos(history_spso, history_ldpso, history_cdpso, scenario_name, max_iter):
    # ... (Logic cũ)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history_spso, label='SPSO (Fixed)', color='blue', linewidth=2)
    ax.plot(history_ldpso, label='L-DPSO (Linear Dynamic)', color='green', linewidth=2, linestyle=':')
    ax.plot(history_cdpso, label='C-DPSO (Chaos Dynamic)', color='red', linewidth=2, linestyle='--')
    ax.set_title(f'Mean Convergence Analysis (N=10) - {scenario_name}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean G_Best Fitness')
    ax.legend()
    ax.grid(True)
    output_filename = f"results/Figure_1_Convergence_3X.pdf"
    plt.savefig(output_filename, format='pdf')
    plt.close(fig)
    print(f"Saved Figure 1 (Convergence Plot).")

# Thêm hàm này vào Runner.py
def debug_max_violation(gbest_pos, N_uavs, N_waypoints, config):
    """ Kiểm tra xem các điểm cuối cùng có vi phạm biên không. """
    path_np = gbest_pos.get().reshape(N_uavs, N_waypoints, 3)
    max_bound = config['sim_params']['max_bound'] # 1000.0

    # Kiểm tra tất cả các điểm trên quỹ đạo
    max_violation = np.max(path_np) - max_bound

    # Kiểm tra chỉ các điểm cuối cùng (W_i, M)
    end_waypoints = path_np[:, -1, :]
    max_end_violation = np.max(end_waypoints) - max_bound

    if max_end_violation > 0.00001:
        print(f"\n[CRITICAL DEBUG] ENDPOINT VIOLATION: Max end value is {np.max(end_waypoints):.8f}")

    return max_violation
# Figure 2: Path Visualization (Giữ nguyên)

def visualize_results(gbest_pos, N_uavs, N_waypoints, config, metrics):

    debug_max_violation(gbest_pos, N_uavs, N_waypoints, config)


    # BẮT BUỘC: Lấy mảng vị trí về CPU và reshape
    path = gbest_pos.get().reshape(N_uavs, N_waypoints, 3)

    # === A. DEBUG: Kiểm tra Dữ liệu Vi phạm ===
    max_val = np.max(path)
    if max_val > 1000.0001: # Giữ cảnh báo này, giờ nó sẽ kiểm tra đúng
        print(f"!!! WARNING: Path max value ({max_val:.4f}) exceeds boundary (1000.0). Check clipping logic.")

    # --- Khởi tạo Figure và Axes ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    # -----------------------------

    # Lấy giới hạn thực tế [X_max, Y_max, Z_max]
    bounds = config['sim_params']['dimensions'] 
    
    # === B. VẼ KHỐI HỘP WIREFRAME (Giữ nguyên logic vẽ 8 đỉnh) ===

    # Định nghĩa 8 đỉnh của khối hộp (Cube/Cuboid)
    X, Y, Z = bounds
    v = np.array([[0,0,0], [0,Y,0], [X,Y,0], [X,0,0],
                  [0,0,Z], [0,Y,Z], [X,Y,Z], [X,0,Z]])

    # Các mặt của khối hộp (để vẽ wireframe)
    faces = [[v[0],v[1],v[2],v[3]], [v[4],v[5],v[6],v[7]],
             [v[0],v[3],v[7],v[4]], [v[1],v[2],v[6],v[5]],
             [v[0],v[1],v[5],v[4]], [v[3],v[2],v[6],v[7]]]

    # Tạo collection (để vẽ wireframe)
    ax.add_collection3d(Poly3DCollection(faces, facecolors=(0,0,0,0.05), 
                                         linewidths=1.5, edgecolors='black', alpha=0.15))

    # Vẽ Chướng ngại vật Tĩnh
    obs_data = np.array(config['obstacles_data'])
    is_first_obs = True
    for i in range(obs_data.shape[0]):
        center = obs_data[i, :3]
        ax.scatter(center[0], center[1], center[2], color='red', marker='o', s=100,
                   label='Static Obstacles' if is_first_obs else None)
        is_first_obs = False

    # Vẽ Mục tiêu Nhiệm vụ
    mission_targets = np.array(config['mission_targets'])
    is_first_target = True
    for target in mission_targets:
        ax.scatter(target[0], target[1], target[2], marker='*', color='gold', s=200,
                   label='Mission Target' if is_first_target else None)
        is_first_target = False

    # Vẽ Đường đi của UAV
    for i in range(N_uavs):
        ax.plot(path[i, :, 0], path[i, :, 1], path[i, :, 2],
                linestyle='-', linewidth=1.5, color='darkslategrey',
                label='UAV Trajectories' if i == 0 else None, alpha=0.7)

        # Điểm Bắt đầu
        start_pos = config['sim_params']['start_pos'][i]
        ax.scatter(start_pos[0], start_pos[1], start_pos[2],
                   marker='s', color='green', s=70, label='Start Position' if i == 0 else None)

        # Điểm Kết thúc
        ax.scatter(path[i, -1, 0], path[i, -1, 1], path[i, -1, 2],
                   marker='x', color='blue', s=70, label='End Waypoint' if i == 0 else None)


    # === C. THIẾT LẬP GIỚI HẠN TRỤC CHẶT CHẼ (Đã loại bỏ padding lớn) ===
    X_max, Y_max, Z_max = bounds[0], bounds[1], bounds[2]
    X_min, Y_min, Z_min = 0, 0, 0 

    # Sử dụng padding nhỏ (1% của giới hạn) để hiển thị viền khối hộp rõ ràng
    soft_padding_xy = X_max * 0.01 
    soft_padding_z = Z_max * 0.01 
    
    ax.set_xlim(X_min - soft_padding_xy, X_max + soft_padding_xy)
    ax.set_ylim(Y_min - soft_padding_xy, Y_max + soft_padding_xy)
    ax.set_zlim(Z_min - soft_padding_z, Z_max + soft_padding_z)
    
    # Thiết lập tỷ lệ box aspect
    ax.set_box_aspect((X_max, Y_max, Z_max))


    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f"{metrics['algorithm']} Optimized Paths - {metrics['scenario']}")

    ax.legend(loc='upper right', fontsize='small')

    output_filename = f"results/Figure_2_{metrics['algorithm']}_best_path.pdf"
    plt.savefig(output_filename, format='pdf')
    plt.close(fig)
    print(f"Saved Figure 2 (Path Visualization).")

# Figure 3: Time Predictability Box Plot (MỚI)
def visualize_time_predictability(stats_spso, stats_ldpso, stats_cdpso, scenario_name):
    """ Tạo Box Plot so sánh sự phân bố thời gian thực thi. """

    # Lấy dữ liệu thời gian thô từ all_metrics
    data_spso = [m['total_time_s'] for m in stats_spso['all_metrics']]
    data_ldpso = [m['total_time_s'] for m in stats_ldpso['all_metrics']]
    data_cdpso = [m['total_time_s'] for m in stats_cdpso['all_metrics']]

    data = [data_spso, data_ldpso, data_cdpso]
    labels = ['SPSO', 'L-DPSO', 'C-DPSO']

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(data, vert=True, patch_artist=True, labels=labels)

    colors = ['#1f77b4', '#2ca02c', '#d62728']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_title(f'Execution Time Predictability (N=10 Runs)')
    ax.set_ylabel('Total Execution Time (seconds)')
    ax.yaxis.grid(True)

    # Thêm chú thích Std Dev
    ax.text(1, np.max(data_spso) + 0.005, f'$\sigma$={stats_spso["Time_StdDev"]:.4f}', ha='center', color='blue')
    ax.text(2, np.max(data_ldpso) + 0.005, f'$\sigma$={stats_ldpso["Time_StdDev"]:.4f}', ha='center', color='green')
    ax.text(3, np.max(data_cdpso) + 0.005, f'$\sigma$={stats_cdpso["Time_StdDev"]:.4f}', ha='center', color='red')

    output_filename = f"results/Figure_3_Time_Predictability_Boxplot.pdf"
    plt.savefig(output_filename, format='pdf')
    plt.close(fig)
    print(f"Saved Figure 3 (Time Predictability Boxplot).")

# Figure 4: Chaos Dynamics (MỚI)
def visualize_chaos_dynamics(W_ldpso, C1_ldpso, C2_ldpso, W_cdpso, C1_cdpso, C2_cdpso, max_iter):

    iterations = range(max_iter + 1) # Từ t=0 đến t=T_max

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # W Comparison
    axs[0].plot(iterations, W_ldpso, label='L-DPSO (Linear)', color='green', linestyle=':')
    axs[0].plot(iterations, W_cdpso, label='C-DPSO (Chaos Modulated)', color='red', alpha=0.7)
    axs[0].set_ylabel('Inertia Weight ($W$)')
    axs[0].set_title('Dynamic Parameter Modulation (Chaos vs Linear)')
    axs[0].legend()
    axs[0].grid(True)

    # C1 Comparison
    axs[1].plot(iterations, C1_ldpso, label='L-DPSO (Linear)', color='green', linestyle=':')
    axs[1].plot(iterations, C1_cdpso, label='C-DPSO (Chaos Modulated)', color='red', alpha=0.7)
    axs[1].set_ylabel('Cognitive Coefficient ($C_1$)')
    axs[1].grid(True)

    # C2 Comparison
    axs[2].plot(iterations, C2_ldpso, label='L-DPSO (Linear)', color='green', linestyle=':')
    axs[2].plot(iterations, C2_cdpso, label='C-DPSO (Chaos Modulated)', color='red', alpha=0.7)
    axs[2].set_ylabel('Social Coefficient ($C_2$)')
    axs[2].set_xlabel('Iteration ($t$)')
    axs[2].grid(True)

    output_filename = f"results/Figure_4_Chaos_Dynamics.pdf"
    plt.savefig(output_filename, format='pdf')
    plt.close(fig)
    print(f"Saved Figure 4 (Chaos Dynamics).")


# --- MAIN EXECUTION ---

if __name__ == "__main__":

    plt.switch_backend('Agg')
    N_RUNS = 10

    final_stats_list = []

    # 0. CHẠY THÍ NGHIỆM HPC SPEEDUP
    hpc_results = time_fitness_evaluation(CONFIG_2)

    # --- PHÂN TÍCH KỊCH BẢN 2 (SPSO vs L-DPSO vs C-DPSO) ---

    # 2.1. SPSO (Fixed)
    config_spso_2 = CONFIG_2.copy()
    config_spso_2['qiso_params']['algo_type'] = "SPSO"
    stats_spso_2 = run_statistical_analysis(config_spso_2, N_RUNS)
    final_stats_list.append(stats_spso_2)

    # 2.2. L-DPSO (Linear Dynamic)
    config_ldpso_2 = CONFIG_2.copy()
    config_ldpso_2['qiso_params']['algo_type'] = "L-DPSO"
    config_ldpso_2['qiso_params']['simulation_name'] = "Scenario_2_LDP"
    stats_ldpso_2 = run_statistical_analysis(config_ldpso_2, N_RUNS)
    final_stats_list.append(stats_ldpso_2)

    # 2.3. C-DPSO (Chaos Dynamic)
    config_cdpso_2 = CONFIG_2.copy()
    config_cdpso_2['qiso_params']['algo_type'] = "C-DPSO"
    config_cdpso_2['qiso_params']['simulation_name'] = "Scenario_2_CDP"
    stats_cdpso_2 = run_statistical_analysis(config_cdpso_2, N_RUNS)
    final_stats_list.append(stats_cdpso_2)


    # --- TẠO FIGURES VÀ TABLES TỪ DỮ LIỆU THU ĐƯỢC ---

    # 1. Figure 1: Hội tụ 3X
    visualize_convergence_comparison_3_algos(
        stats_spso_2['Convergence_Mean'],
        stats_ldpso_2['Convergence_Mean'],
        stats_cdpso_2['Convergence_Mean'],
        "Scenario_2",
        CONFIG_2['qiso_params']['max_iter']
    )

    # 2. Figure 2: Trực quan hóa đường đi tốt nhất (3 files riêng biệt)
    visualize_results(stats_spso_2['Best_Run_Pos'], stats_spso_2['N_uavs'], stats_spso_2['N_waypoints'], CONFIG_2, {'algorithm': 'SPSO', 'scenario': 'Scenario_2'})
    visualize_results(stats_ldpso_2['Best_Run_Pos'], stats_ldpso_2['N_uavs'], stats_ldpso_2['N_waypoints'], CONFIG_2, {'algorithm': 'L-DPSO', 'scenario': 'Scenario_2_LDP'})
    visualize_results(stats_cdpso_2['Best_Run_Pos'], stats_cdpso_2['N_uavs'], stats_cdpso_2['N_waypoints'], CONFIG_2, {'algorithm': 'C-DPSO', 'scenario': 'Scenario_2_CDP'})


    # 3. Figure 3: Phân bố thời gian (Box Plot)
    visualize_time_predictability(stats_spso_2, stats_ldpso_2, stats_cdpso_2, "Scenario_2")

    # 4. Figure 4: Động lực học Tham số (So sánh L-DPSO và C-DPSO)
    visualize_chaos_dynamics(
        stats_ldpso_2['W_hist_sample'], stats_ldpso_2['C1_hist_sample'], stats_ldpso_2['C2_hist_sample'],
        stats_cdpso_2['W_hist_sample'], stats_cdpso_2['C1_hist_sample'], stats_cdpso_2['C2_hist_sample'],
        CONFIG_2['qiso_params']['max_iter']
    )


    # 5. Lưu 3 Tables CSV
    save_summary_csv(final_stats_list, hpc_results)
