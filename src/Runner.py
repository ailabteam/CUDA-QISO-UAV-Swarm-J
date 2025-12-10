# src/Runner.py (Phiên bản Cuối cùng với Phân tích Thống kê)

import cupy as cp
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import csv
import random

# Tăng tính ngẫu nhiên của các lần chạy
random.seed(42)
np.random.seed(42)

from src.Environment import UAV_Environment
from src.QISO_Core import QISO_Optimizer
from data.config_scenario1 import SCENARIO_CONFIG as CONFIG_1
from data.config_scenario2 import SCENARIO_CONFIG_2 as CONFIG_2

# --- CÁC HÀM XỬ LÝ LÕI VÀ LƯU TRỮ ---

def run_single_simulation(config, seed_val):
    
    # Thiết lập seed cho CuPy và NumPy/Python
    cp.random.seed(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)
    
    qiso_params = config['qiso_params']
    algo_type = "QISO" if qiso_params.get('is_qiso', True) else "SPSO"
    qiso_params['algo_type'] = algo_type
    
    # Không in log tiến trình cho các lần chạy lặp lại
    # print(f"-> Running {algo_type} (Seed: {seed_val})...", end='\r')
    
    env = UAV_Environment(
        N_uavs=config['sim_params']['N_uavs'], 
        N_waypoints=config['sim_params']['N_waypoints'], 
        config=config, 
        cp=cp
    )
    
    optimizer = QISO_Optimizer(
        env=env, 
        N_particles=qiso_params['N_particles'], 
        max_iter=qiso_params['max_iter'], 
        params=qiso_params, 
        cp=cp
    )
    
    start_time = time.time()
    gbest_position, gbest_fitness, history = optimizer.optimize() 
    end_time = time.time()
    
    total_time = end_time - start_time
    
    metrics = {
        "seed": seed_val,
        "algorithm": algo_type,
        "scenario": qiso_params['simulation_name'],
        "gbest_fitness": float(gbest_fitness),
        "total_time_s": total_time,
        "convergence_history": history
    }
    
    return gbest_position, metrics

def run_statistical_analysis(config, N_RUNS=10):
    
    algo_type = "QISO" if config['qiso_params'].get('is_qiso', True) else "SPSO"
    scenario_name = config['qiso_params']['simulation_name']
    
    print(f"\n======== Running Statistical Test: {scenario_name} [{algo_type}] (N={N_RUNS}) ========")
    
    all_metrics = []
    best_run_fitness = np.inf
    best_pos = None
    
    # Khởi tạo ma trận để lưu trữ lịch sử hội tụ (N_runs x Max_iter)
    conv_matrix = np.zeros((N_RUNS, config['qiso_params']['max_iter']))
    
    for i in range(N_RUNS):
        # Sử dụng seed dựa trên i và seed cố định ban đầu
        current_seed = 42 + i * 10 
        
        # Chạy simulation
        gbest_pos, metrics = run_single_simulation(config, current_seed)
        
        all_metrics.append(metrics)
        conv_matrix[i, :] = metrics['convergence_history']
        
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
        "N_waypoints": config['sim_params']['N_waypoints']
    }
    
    print("\n--- Statistical Summary ---")
    print(f"Fitness: {stats['Fitness_Mean']:.2f} ± {stats['Fitness_StdDev']:.2f} (Min: {stats['Fitness_Min']:.2f})")
    print(f"Time: {stats['Time_Mean']:.2f}s ± {stats['Time_StdDev']:.2f}s")
    
    return stats


def save_summary_csv(stats_list, filename="statistical_summary.csv"):
    """ Lưu tóm tắt thống kê của tất cả các kịch bản vào một file CSV. """
    
    filepath = f"results/{filename}"
    
    fieldnames = ['Scenario', 'Algorithm', 'N_runs', 
                  'Fitness_Mean', 'Fitness_StdDev', 'Fitness_Min', 
                  'Time_Mean', 'Time_StdDev', 'N_uavs', 'N_waypoints']
    
    if not os.path.exists(filepath):
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        for stats in stats_list:
            row = {
                'Scenario': stats['scenario'],
                'Algorithm': stats['algorithm'],
                'N_runs': stats['N_runs'],
                'Fitness_Mean': f"{stats['Fitness_Mean']:.4f}",
                'Fitness_StdDev': f"{stats['Fitness_StdDev']:.4f}",
                'Fitness_Min': f"{stats['Fitness_Min']:.4f}",
                'Time_Mean': f"{stats['Time_Mean']:.4f}",
                'Time_StdDev': f"{stats['Time_StdDev']:.4f}",
                'N_uavs': stats['N_uavs'],
                'N_waypoints': stats['N_waypoints']
            }
            writer.writerow(row)
    print(f"\nSaved statistical summary to {filepath}")


# --- CÁC HÀM VISUALIZATION (Đã sửa Legend và thêm Convergence Plot) ---
# ... (visualize_results giữ nguyên logic từ Bước 8 - đã có legend) ...

def visualize_results(gbest_pos, N_uavs, N_waypoints, config, metrics):
    # ... (Hàm này giữ nguyên logic từ Bước 8/9 - đã được confirm là có Legend) ...
    path = gbest_pos.reshape(N_uavs, N_waypoints, 3)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    obs_data = np.array(config['obstacles_data'])
    is_first_obs = True
    for i in range(obs_data.shape[0]):
        center = obs_data[i, :3]
        ax.scatter(center[0], center[1], center[2], color='red', marker='o', s=100, 
                   label='Static Obstacles' if is_first_obs else None)
        is_first_obs = False

    mission_targets = np.array(config['mission_targets'])
    is_first_target = True
    for target in mission_targets:
        ax.scatter(target[0], target[1], target[2], marker='*', color='gold', s=200,
                   label='Mission Target' if is_first_target else None)
        is_first_target = False
    
    for i in range(N_uavs):
        ax.plot(path[i, :, 0], path[i, :, 1], path[i, :, 2], 
                linestyle='-', linewidth=1.5, label=f'UAV {i+1}' if i == 0 else None, alpha=0.7)
        
        start_pos = config['sim_params']['start_pos'][i]
        ax.scatter(start_pos[0], start_pos[1], start_pos[2], 
                   marker='s', color='green', s=70, label='Start Position' if i == 0 else None)
        
        ax.scatter(path[i, -1, 0], path[i, -1, 1], path[i, -1, 2], 
                   marker='x', color='blue', s=70, label='End Waypoint' if i == 0 else None)
    
    bounds = config['sim_params']['dimensions']
    ax.set_xlim(0, bounds[0])
    ax.set_ylim(0, bounds[1])
    ax.set_zlim(0, bounds[2])
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f"{metrics['algorithm']} Optimized Paths - {metrics['scenario']}")
    
    ax.legend(loc='upper right', fontsize='small')
    
    output_filename = f"results/{metrics['scenario']}_{metrics['algorithm']}_best_path.pdf"
    plt.savefig(output_filename, format='pdf')
    plt.close(fig) 
    print(f"Saved visualization (Best Run) to {output_filename}")


def visualize_convergence_comparison(history_spso, history_qiso, scenario_name, max_iter):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Lấy giá trị trung bình từ list của numpy array
    ax.plot(history_spso, label='SPSO (Baseline)', color='blue', linewidth=2)
    ax.plot(history_qiso, label='QISO-Dynamic (Proposed)', color='red', linewidth=2, linestyle='--')
    
    ax.set_title(f'Mean Convergence Analysis (N=10) - {scenario_name}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean G_Best Fitness')
    ax.legend()
    ax.grid(True)
    
    output_filename = f"results/{scenario_name}_mean_convergence.pdf"
    plt.savefig(output_filename, format='pdf')
    plt.close(fig)
    print(f"Saved mean convergence plot to {output_filename}")


# --- MAIN EXECUTION ---

if __name__ == "__main__":
    
    plt.switch_backend('Agg') 
    
    # Số lần chạy thống kê (có thể tăng lên 30 cho paper)
    N_RUNS = 10 
    
    # Khởi tạo list lưu trữ kết quả thống kê
    final_stats_list = []
    
    # --- PHÂN TÍCH KỊCH BẢN 1 (Đã dùng CONFIG_1 từ data) ---
    
    # 1.1. SPSO trên KỊCH BẢN 1
    config_spso_1 = CONFIG_1.copy()
    config_spso_1['qiso_params']['is_qiso'] = False
    config_spso_1['qiso_params']['simulation_name'] = "Scenario_1"
    stats_spso_1 = run_statistical_analysis(config_spso_1, N_RUNS)
    final_stats_list.append(stats_spso_1)
    
    # 1.2. QISO trên KỊCH BẢN 1
    config_qiso_1 = CONFIG_1.copy()
    config_qiso_1['qiso_params']['is_qiso'] = True
    config_qiso_1['qiso_params']['simulation_name'] = "Scenario_1"
    stats_qiso_1 = run_statistical_analysis(config_qiso_1, N_RUNS)
    final_stats_list.append(stats_qiso_1)
    
    # Trực quan hóa hội tụ Kịch bản 1 (sử dụng mean history)
    visualize_convergence_comparison(stats_spso_1['Convergence_Mean'], 
                                     stats_qiso_1['Convergence_Mean'], 
                                     "Scenario_1_Comparison", 
                                     CONFIG_1['qiso_params']['max_iter'])
    
    # Trực quan hóa quỹ đạo tốt nhất (Best Run)
    visualize_results(stats_spso_1['Best_Run_Pos'], stats_spso_1['N_uavs'], stats_spso_1['N_waypoints'], CONFIG_1, {'algorithm': 'SPSO', 'scenario': 'Scenario_1'})
    visualize_results(stats_qiso_1['Best_Run_Pos'], stats_qiso_1['N_uavs'], stats_qiso_1['N_waypoints'], CONFIG_1, {'algorithm': 'QISO', 'scenario': 'Scenario_1'})
    
    
    # --- PHÂN TÍCH KỊCH BẢN 2 (Đã dùng CONFIG_2 từ data) ---
    
    # 2.1. SPSO trên KỊCH BẢN 2
    config_spso_2 = CONFIG_2.copy()
    config_spso_2['qiso_params']['is_qiso'] = False
    config_spso_2['qiso_params']['simulation_name'] = "Scenario_2"
    stats_spso_2 = run_statistical_analysis(config_spso_2, N_RUNS)
    final_stats_list.append(stats_spso_2)

    # 2.2. QISO trên KỊCH BẢN 2
    config_qiso_2 = CONFIG_2.copy()
    config_qiso_2['qiso_params']['is_qiso'] = True
    config_qiso_2['qiso_params']['simulation_name'] = "Scenario_2"
    stats_qiso_2 = run_statistical_analysis(config_qiso_2, N_RUNS)
    final_stats_list.append(stats_qiso_2)
    
    # Trực quan hóa hội tụ Kịch bản 2 (sử dụng mean history)
    visualize_convergence_comparison(stats_spso_2['Convergence_Mean'], 
                                     stats_qiso_2['Convergence_Mean'], 
                                     "Scenario_2_Comparison", 
                                     CONFIG_2['qiso_params']['max_iter'])

    # Trực quan hóa quỹ đạo tốt nhất (Best Run)
    visualize_results(stats_spso_2['Best_Run_Pos'], stats_spso_2['N_uavs'], stats_spso_2['N_waypoints'], CONFIG_2, {'algorithm': 'SPSO', 'scenario': 'Scenario_2'})
    visualize_results(stats_qiso_2['Best_Run_Pos'], stats_qiso_2['N_uavs'], stats_qiso_2['N_waypoints'], CONFIG_2, {'algorithm': 'QISO', 'scenario': 'Scenario_2'})


    # --- LƯU TRỮ KẾT QUẢ CUỐI CÙNG VÀO CSV ---
    save_summary_csv(final_stats_list)
