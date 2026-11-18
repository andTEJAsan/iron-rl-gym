"""
JCB 8026 CTS Hardware, Geometry & Control Configuration (v3 — Drone-Free MVP + VLM Supervisor)
==============================================================================================
Incorporates all collaborator review notes + physical deployment specs:
  1. Drone Dropped (Cab-Only Sensor Suite):
     - C1: Livox Mid360S 3D LiDAR (upside-down under cab roof front plate)
     - C3: Reolink Duo 2 PoE 180° Dual-Lens Camera (on cab roof looking along boom + truck)
     - C4: 4x WitMotion WT901WIFI 6-axis IMUs (Boom, Dipper, Bucket link, Cab yaw)
     - `task_vec` (8D) is now derived 100% in the Cab-Ego Frame from C1/C3/C4 + VLM Supervisor!
  2. Exact JCB 8026 CTS Spec-Sheet Geometry & Tearout Limits:
     - Operating Weight: 2,867 kg
     - Track Length: 1.970 m | Width Over Tracks: 1.550 m | Shoe Width: 0.300 m | Tail Swing: 1.145 m
     - Cab Roof Height: 2.400 m | Ground Clearance: 0.300 m
     - Boom Length (pin-to-pin): 2.100 m | Dipper Length: 1.300 m | Bucket Capacity: 0.08 m^3
     - Max Dig Reach: 4.770 m | Max Dig Depth: 2.740 m | Max Dump Height: 3.220 m
     - Bucket Tearout Force: 22.9 kN | Dipper Tearout Force: 14.6 kN
  3. 50 Hz ELRS 4-Channel ISO Joystick Mapping (Section 3.3):
     - Ch 1 (idx 0): Left Fore-Aft     -> Dipper (Stick)
     - Ch 2 (idx 1): Left Side-to-Side -> Slew (Cab Swing)
     - Ch 3 (idx 2): Right Fore-Aft    -> Boom
     - Ch 4 (idx 3): Right Side-to-Side-> Bucket Curl / Dump
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np


@dataclass(frozen=True)
class JCB8026Config:
    # --- Exact JCB 8026 CTS Official Spec-Sheet Dimensions (meters & kN) ---
    operating_weight_kg: float = 2867.0
    track_length_m: float = 1.970
    track_width_m: float = 1.550
    track_shoe_width_m: float = 0.300
    tail_swing_radius_m: float = 1.145
    cab_roof_height_m: float = 2.400
    ground_clearance_m: float = 0.300
    boom_length_m: float = 2.100
    dipper_length_m: float = 1.300
    bucket_width_m: float = 0.500
    bucket_capacity_m3: float = 0.080
    max_dig_reach_m: float = 4.770
    max_dig_depth_m: float = 2.740
    max_dump_height_m: float = 3.220
    bucket_tearout_kn: float = 22.9
    dipper_tearout_kn: float = 14.6

    # --- Control Loop & ELRS Radio (Section 2 & 3) ---
    control_hz: int = 50
    dt: float = 0.02  # 20 ms tick
    action_dim: int = 4

    # Exact ISO Channel Order (0-indexed in Python tensors, Ch 1..4 on ER5C V2 receiver)
    channel_names: Tuple[str, str, str, str] = (
        "ch1_left_fore_aft_dipper",
        "ch2_left_side_slew",
        "ch3_right_fore_aft_boom",
        "ch4_right_side_bucket",
    )

    # PWM / CRSF limits (microseconds)
    pwm_min_us: int = 1000
    pwm_neutral_us: int = 1500
    pwm_max_us: int = 2000
    joystick_deadband: float = 0.04  # 4% mechanical pilot valve deadband

    # ANNIMOS 60KG DS5160SSG @ 7.4V (Hobbywing 10A UBEC) max stick slew rate
    servo_max_rate_per_sec: float = 5.5

    # --- JCB 8026 Joint Limits (radians, matching WitMotion WT901WIFI 6-axis calibration) ---
    joint_limits_rad: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "dipper": (-1.50, 0.40),   # Ch 1: Left Fore-Aft
            "slew":   (-1.60, 1.60),   # Ch 2: Left Side
            "boom":   (-1.05, 0.55),   # Ch 3: Right Fore-Aft
            "bucket": (-1.45, 0.95),   # Ch 4: Right Side
        }
    )

    max_joint_vel_rad_s: Tuple[float, float, float, float] = (
        1.65,  # Ch 1: Dipper
        1.35,  # Ch 2: Slew
        1.45,  # Ch 3: Boom
        2.10,  # Ch 4: Bucket
    )

    # --- Cab-Only Sensor Suite Tensor Shapes (No Drone!) ---
    # C4: 4x WitMotion WT901WIFI (angles + velocities) + 4x last stick action = 12D proprio
    proprio_dim: int = 12
    # Cab-derived target vector (from C1 Livox + C3 Reolink 180° + C4 Cab Slew IMU + VLM Supervisor):
    # [dig_dx_cab, dig_dy_cab, dig_dz_cab, truck_dx_cab, truck_dy_cab, truck_dz_cab, cycle_phase_progress, bucket_fill_est]
    task_vec_dim: int = 8
    # C1: Livox Mid360S upside-down cab-roof 3D LiDAR projected to 2.5D local elevation map
    elevation_shape: Tuple[int, int, int] = (1, 32, 32)
    # C3: Reolink Duo 2 PoE cab-roof camera crop along the boom
    cab_rgb_shape: Tuple[int, int, int] = (3, 64, 64)

    # --- Safety Watchdog (Section 5) ---
    sensor_staleness_timeout_sec: float = 0.20  # 200 ms max age before neutral failsafe
    elrs_serial_baud: int = 420000


CONFIG = JCB8026Config()


def sticks_to_pwm_us(sticks: np.ndarray, deadman_active: bool = True) -> np.ndarray:
    """
    Convert normalized stick values [-1, +1]^4 in ISO channel order
    [Ch1 Dipper, Ch2 Slew, Ch3 Boom, Ch4 Bucket] into ER5C V2 PWM microseconds [1000..2000].
    Always outputs neutral [1500, 1500, 1500, 1500] if deadman_active is False.
    """
    if not deadman_active:
        return np.full(4, CONFIG.pwm_neutral_us, dtype=np.int32)
    clipped = np.clip(np.asarray(sticks, dtype=np.float32), -1.0, 1.0)
    pwm = CONFIG.pwm_neutral_us + clipped * 500.0
    return np.rint(pwm).astype(np.int32)


def sticks_to_crsf_frame(sticks: np.ndarray, deadman_active: bool = True) -> bytes:
    """
    Encode the 4 ISO joystick channels (plus neutral Aux channels 5..16) into a standard
    26-byte ExpressLRS / CRSF RC_CHANNELS_PACKED frame (0xC8 sync, 0x16 type).
    """
    pwm = sticks_to_pwm_us(sticks, deadman_active=deadman_active)
    ticks = [992] * 16
    for i in range(4):
        frac = (float(pwm[i]) - 1000.0) / 1000.0
        ticks[i] = int(np.clip(round(172 + frac * (1811 - 172)), 172, 1811))

    bit_buf = 0
    for ch_idx, val in enumerate(ticks):
        bit_buf |= (val & 0x7FF) << (11 * ch_idx)
    payload = bit_buf.to_bytes(22, byteorder="little")

    frame_type = bytes([0x16])
    body = frame_type + payload
    crc = 0
    for b in body:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0xD5) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF

    return bytes([0xC8, 24]) + body + bytes([crc])
