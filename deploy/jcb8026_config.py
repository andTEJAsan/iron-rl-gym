"""
JCB 8026 CTS Hardware & Control Configuration (Working Backwards from Deployment)
=================================================================================
Matches the physical "Two-week demo: hardware list" specification:
- Machine: JCB 8026 CTS Compact Excavator (~2.7t, boom 2.1m, dipper 1.25m, bucket 0.08m^3)
- Control Rate: 50 Hz (20 ms loop period)
- Radio Link: Adafruit FTDI Friend (USB-Serial) -> Radiomaster Ranger Micro 2.4GHz ELRS
              -> Radiomaster ER5C V2 2.4GHz ELRS Receiver -> 4x ANNIMOS 60KG DS5160SSG Servos
- ISO 4-Channel Joystick Mapping (Section 3.3):
    Channel 1 (idx 0): Left Fore-Aft     -> Dipper (Stick)
    Channel 2 (idx 1): Left Side-to-Side -> Slew (Swing)
    Channel 3 (idx 2): Right Fore-Aft    -> Boom
    Channel 4 (idx 3): Right Side-to-Side-> Bucket Curl
- Safety / Failsafe (Section 2 & 5):
    - Deadman switch on PowerA Wired Xbox Controller (Right Trigger RT > 0.5)
    - Sensor staleness watchdog (200 ms timeout on WitMotion UDP, Livox Mid360S, Reolink RTSP)
    - Neutral command: [0.0, 0.0, 0.0, 0.0] -> [1500, 1500, 1500, 1500] us PWM failsafe
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np


@dataclass(frozen=True)
class JCB8026Config:
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
    servo_max_rate_per_sec: float = 5.5  # full [-1, 1] sweep in ~0.36s

    # --- JCB 8026 Joint Limits (radians, matching WitMotion WT901WIFI 6-axis calibration) ---
    # Joint order in MuJoCo qpos mirrors the 4 ISO channels: [dipper, slew, boom, bucket]
    joint_limits_rad: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "dipper": (-1.50, 0.40),   # Ch 1: Left Fore-Aft (Stick extend / crowd)
            "slew":   (-1.60, 1.60),   # Ch 2: Left Side (Cab swing left / right)
            "boom":   (-1.05, 0.55),   # Ch 3: Right Fore-Aft (Boom lower / raise)
            "bucket": (-1.45, 0.95),   # Ch 4: Right Side (Bucket curl / dump)
        }
    )

    # Maximum hydraulic angular velocity per joint (rad/s) at low/demo engine RPM
    max_joint_vel_rad_s: Tuple[float, float, float, float] = (
        1.65,  # Ch 1: Dipper
        1.35,  # Ch 2: Slew
        1.45,  # Ch 3: Boom
        2.10,  # Ch 4: Bucket
    )

    # --- Sensor Suite Tensor Shapes (Section 4) ---
    # C4: 4x WitMotion WT901WIFI (angles + velocities) + 4x last stick action = 12D proprio
    proprio_dim: int = 12
    # A1/F1: DJI Mini 4 Pro overhead AprilTag relative target vector:
    # [trench_dx, trench_dy, trench_dz, truck_dx, truck_dy, truck_dz, truck_dyaw, bucket_load_est] = 8D goal/task state
    task_vec_dim: int = 8
    # C1: Livox Mid360S upside-down cab-roof 3D LiDAR projected to 2.5D local elevation map
    elevation_shape: Tuple[int, int, int] = (1, 32, 32)
    # C3: Reolink Duo 2 PoE cab-roof camera crop along the boom
    cab_rgb_shape: Tuple[int, int, int] = (3, 64, 64)

    # --- Safety Watchdog (Section 5) ---
    sensor_staleness_timeout_sec: float = 0.20  # 200 ms max age before neutral failsafe
    elrs_serial_baud: int = 420000  # Standard CRSF baud rate over Adafruit FTDI Friend


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
    26-byte ExpressLRS / CRSF RC_CHANNELS_PACKED frame (0xC8 sync, 0x16 type) compatible
    with Radiomaster Ranger Micro (B4) via Adafruit FTDI Friend (B5) / kaack/elrs-joystick-control.
    """
    pwm = sticks_to_pwm_us(sticks, deadman_active=deadman_active)
    # Map 1000..2000 us -> CRSF 11-bit ticks (172..1811, center 992)
    ticks = [992] * 16
    for i in range(4):
        frac = (float(pwm[i]) - 1000.0) / 1000.0
        ticks[i] = int(np.clip(round(172 + frac * (1811 - 172)), 172, 1811))

    # Pack 16 x 11-bit channels into 22 bytes (little-endian bitstream)
    bit_buf = 0
    for ch_idx, val in enumerate(ticks):
        bit_buf |= (val & 0x7FF) << (11 * ch_idx)
    payload = bit_buf.to_bytes(22, byteorder="little")

    frame_type = bytes([0x16])  # CRSF_FRAMETYPE_RC_CHANNELS_PACKED
    body = frame_type + payload
    # Simple CRC8 (poly 0xD5) used by CRSF
    crc = 0
    for b in body:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0xD5) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF

    # [Address=0xC8, Length=24, Type=0x16, 22-byte payload, CRC8]
    return bytes([0xC8, 24]) + body + bytes([crc])
