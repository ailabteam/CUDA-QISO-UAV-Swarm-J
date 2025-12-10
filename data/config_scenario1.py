# data/config_scenario1.py

# --- Tham số Môi trường Tổng quan (SIM_PARAMS) ---
SIM_PARAMS = {
    "dimensions": [1000, 1000, 500],  # LxWxH (m)
    "min_bound": 0.0,
    "max_bound": 1000.0,
    "N_uavs": 4,                     # Số lượng UAV trong đàn
    "N_waypoints": 10,               # Số điểm tham chiếu cho mỗi UAV (M)
    "weights": [1.0, 50.0, 10.0],    # Trọng số [f1: Thời gian, f2: Va chạm, f3: Nhiệm vụ]
    "start_pos": [[10, 10, 50], [10, 20, 50], [20, 10, 50], [20, 20, 50]], # Vị trí khởi đầu
}

# --- Dữ liệu Chướng ngại vật Tĩnh (STATIC_OBSTACLES) ---
# Format: [x_center, y_center, z_center, radius_safe_zone]
STATIC_OBSTACLES = [
    [200, 200, 100, 50],
    [750, 800, 250, 70],
    [100, 900, 300, 40],
    [900, 100, 150, 60]
]

# --- Điểm Nhiệm vụ (MISSION_TARGETS) ---
MISSION_TARGETS = [
    [500, 500, 200, 30], # (x, y, z, required_radius)
    [800, 200, 350, 30],
    [200, 800, 150, 30]
]

# --- Tham số QISO (QISO_PARAMS) ---
QISO_PARAMS = {
    "N_particles": 200,              
    "max_iter": 500,                 
    "C1": 1.5,                       
    "C2": 1.5,                       
    "L_factor": 0.5,                 
    "simulation_name": "Scenario_1_Static"
}

# Kết hợp tất cả thành SCENARIO_CONFIG duy nhất
SCENARIO_CONFIG = {
    "sim_params": SIM_PARAMS,
    "obstacles_data": STATIC_OBSTACLES,
    "mission_targets": MISSION_TARGETS,
    "qiso_params": QISO_PARAMS # Đặt QISO_PARAMS ở đây để dễ dàng truy cập
}
