# 🚗 Autonomous Parallel Parking in CARLA

#Overview

This project implements an **autonomous parallel parking system** in the CARLA Simulator using a **rule-based control approach**.

The system detects parking slots using LiDAR data and executes a complete parking maneuver using a **Finite State Machine (FSM)** and handcrafted control strategies.

Unlike learning-based methods, this approach is:
* Lightweight
* Interpretable
* Real-time capable



#Key Features

**Autonomous parallel parking**
**LiDAR-based slot detection**
**Finite State Machine (FSM) control**
**Safety-aware motion using obstacle detection**
**No ROS required (pure Python implementation)**


#System Architecture

The system consists of the following components:
* **Slot Detector**
  * Processes LiDAR point cloud
  * Identifies valid parking gaps

* **Finite State Machine (FSM)**

  * Controls the parking maneuver:
    * Scanning
    * Forward alignment
    * Reverse (Phase 1 & 2)
    * Final centering

* **Safety Monitor**
  * Prevents collisions using LiDAR-based clearance checks

* **Controller**
  * Speed PID + steering logic


## FSM Workflow

SCANNING → FORWARD ALIGN → STOP → REVERSE (Phase 1) → REVERSE (Phase 2) → CENTERING → PARKED


#Requirements
* Python 3.8+
* CARLA 0.9.15
* NumPy
* OpenCV

#How to Run

1. Start CARLA: CarlaUE4.exe
2. Open Anaconda Prompt
3. go to the location of your .py file
4. activate carla-sim
5. Run the script: Autonomous_Parking_CARLA.py




#Key Parameters

| Parameter         | Description             |
| ----------------- | ----------------------- |
| `MIN_SLOT_LENGTH` | Minimum gap for parking |
| `TARGET_EX_TOL`   | Longitudinal tolerance  |
| `TARGET_EY_TOL`   | Lateral tolerance       |
| `TARGET_YAW_TOL`  | Orientation tolerance   |



#Results

The system successfully:
* Detects parking slots using real-time LiDAR
* Performs smooth multi-stage parking
* Avoids collisions using safety constraints


#Methodology

This project uses a **rule-based approach** combining:
* Geometric reasoning
* Sensor-based perception
* Finite State Machine control

This avoids:
* Long training times
* High computational cost of deep learning


#Future Improvements

* Improve trajectory optimization (MPC)
* Extend to perpendicular parking
* Integrate learning-based perception


#Author
**Salma Diaa**
PhD Student – Smart Cities
Western University

