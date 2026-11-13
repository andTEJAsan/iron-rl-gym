# IronBench-8026: Autonomous Digging & Truck-Loading Benchmark Leaderboard

Standardized evaluation across **5 MuJoCo 3D gravel excavation scenarios (`T1_Nominal`..`T5_Sim2Real_Stress`)** using the exact **JCB 8026 CTS** 4-channel ISO servo interface (`[Ch1 Dipper, Ch2 Slew, Ch3 Boom, Ch4 Bucket]`).

## 1. Overall Benchmark Scorecard (Mean Across T1–T5)

| Policy / Stage | IronScore (0–100) | Truck Load Success (%) | Throughput (rocks/min) | Bucket Fill Ratio (%) | Spillage Rate (%) | Servo Smoothness TV (down) | Mean Episode Return |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random** | **1.4** | 0.0% | 0.0 | 0.0% | 100.0% | `0.1725` | 20.7 |
| **Stage 1: BC (Imitation)** | **74.1** | 87.1% | 41.7 | 60.0% | 34.7% | `0.0317` | 1108.4 |
| **Stage 2: RL Fine-Tuned** | **88.4** | 100.0% | 62.3 | 82.2% | 31.8% | `0.0121` | 2015.3 |
| **Scripted Oracle** | **80.7** | 85.7% | 40.6 | 78.3% | 36.2% | `0.0116` | 1944.0 |

## 2. Per-Scenario Truck Loading Throughput (Boulders Delivered / Min)

| Policy / Stage | `T1_Nominal` | `T2_DeepTrench` | `T3_RandomTruckPose` | `T4_HeavyWetSoil` | `T5_Sim2Real_Stress` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **Stage 1: BC (Imitation)** | 37.1 | 45.7 | 54.3 | 34.3 | 37.1 |
| **Stage 2: RL Fine-Tuned** | 65.7 | 57.1 | 71.4 | 60.0 | 57.1 |
| **Scripted Oracle** | 60.0 | 31.4 | 37.1 | 37.1 | 37.1 |

## 3. 50 Hz ELRS Deployment Node Verification (`deploy/elrs_control_node.py`)

- **50 Hz Ticks Executed:** `200` (`20 ms` cycle)
- **Shadow Mode Mean L1 Stick Divergence (`||a_model - a_human||_1`):** `0.0453`
- **Deadman Release & 200 ms Sensor Staleness Failsafe (`1500 µs` across Ch 1–4):** `PASS`
- **Sample 26-Byte ExpressLRS CRSF Packet (Hex):** `c8181618fba25679063ef0810f7ce0031ff8c0073ef0810f7cb2`
