# data/config_scenario2.py

# Kịch bản phức tạp hơn: Nhiều UAV, thêm chướng ngại vật động
SIM_PARAMS_2 = {
    "dimensions": [1000, 1000, 500],
    "min_bound": 0.0,
    "max_bound": 1000.0,
    "N_uavs": 8,                     # Tăng số lượng UAV lên 8
    "N_waypoints": 15,               # Tăng số waypoint
    "weights": [1.0, 100.0, 10.0],   # Tăng trọng số phạt va chạm (f2)
    "min_separation": 10.0,          # Khoảng cách an toàn tối thiểu giữa các UAV (m)
    "start_pos": [
        [50, 50, 50], [50, 100, 50], [100, 50, 50], [100, 100, 50], 
        [50, 950, 50], [50, 900, 50], [100, 950, 50], [100, 900, 50]
    ]
}

# Thêm nhiều chướng ngại vật tĩnh
STATIC_OBSTACLES_2 = [
    [200, 200, 100, 50], [750, 800, 250, 70], [100, 900, 300, 40], [900, 100, 150, 60],
    [500, 500, 150, 80],
    [300, 600, 50, 40],
    [650, 350, 300, 55]
]

MISSION_TARGETS_2 = [
    [500, 500, 200, 30], [800, 200, 350, 30], [200, 800, 150, 30],
    [400, 100, 100, 20],
    [900, 400, 400, 20]
]

QISO_PARAMS_2 = {
    "N_particles": 500,              
    "max_iter": 700,                 
    "C1": 1.5,
    "C2": 1.5,
    "L_factor": 0.5,
    "W": 0.7,                        
    "simulation_name": "Scenario_2_Dynamic_Swarm",
    "is_qiso": True
}

SCENARIO_CONFIG_2 = {
    "sim_params": SIM_PARAMS_2,
    "obstacles_data": STATIC_OBSTACLES_2,
    "mission_targets": MISSION_TARGETS_2,
    "qiso_params": QISO_PARAMS_2
}
