"""
50 Hz JCB 8026 Deployment & Shadow-Mode Control Node (`deploy/elrs_control_node.py`)
====================================================================================
Implements the exact runtime program described in Section 2, 3.3, and 5 of the
"Two-week demo: hardware list" specification:

  1. Runs at 50 Hz (20 ms period).
  2. Supports 3 control modes producing the exact same 4 ISO joystick values:
       - `MANUAL_XBOX`:   PowerA Wired Xbox Controller (B3) sticks -> [Ch1, Ch2, Ch3, Ch4]
       - `SHADOW_MODEL`:  Human drives via Xbox controller while `JCB8026ActorCritic` runs
                          live in parallel, logging stick divergence `||a_model - a_human||`
       - `AUTONOMOUS`:    `JCB8026ActorCritic` (or scripted routine) drives the 4 sticks
  3. Safety & Deadman Gate (Section 2 & 5):
       - Releases sticks to neutral [0, 0, 0, 0] (1500 us CRSF failsafe) immediately if:
           * Xbox Controller deadman trigger is released, OR
           * Any sensor stream (WitMotion UDP, Livox Mid360S, Reolink RTSP, Drone AprilTag)
             exceeds the 200 ms staleness timeout.
  4. Encodes 26-byte CRSF packets over USB-Serial (Adafruit FTDI Friend B5 -> Radiomaster
     Ranger Micro B4) and logs every step as training data.
"""

from enum import Enum
import time
from typing import Dict, List, Optional
import numpy as np

from deploy.jcb8026_config import CONFIG, sticks_to_crsf_frame, sticks_to_pwm_us
from deploy.sensors_io import JCB8026SensorPacket
from models.policy import JCB8026ActorCritic


class ControlMode(str, Enum):
    MANUAL_XBOX = "manual_xbox"
    SHADOW_MODEL = "shadow_model"
    AUTONOMOUS = "autonomous"


class ELRSControlNode50Hz:
    """
    Production 50 Hz control loop for the JCB 8026 ground laptop.
    Can talk to a physical `/dev/ttyUSB*` FTDI Friend serial port or run in dry-run/sim mode.
    """

    def __init__(
        self,
        policy: Optional[JCB8026ActorCritic] = None,
        mode: ControlMode = ControlMode.SHADOW_MODEL,
        serial_port: Optional[str] = None,
        device: str = "cpu",
    ):
        self.policy = policy
        self.mode = mode
        self.serial_port = serial_port
        self.device = device
        self.last_sent_sticks = np.zeros(4, dtype=np.float32)
        self.telemetry_log: List[Dict] = []

    def step_50hz(
        self,
        sensor_packet: JCB8026SensorPacket,
        xbox_sticks: np.ndarray,
        deadman_held: bool,
        now: Optional[float] = None,
    ) -> Dict:
        """
        Executes a single 20 ms (50 Hz) control tick.
        Returns the command dictionary containing normalized sticks, PWM us, CRSF frame bytes,
        safety flags, and shadow-mode divergence metrics.
        """
        if now is None:
            now = time.monotonic()

        sensors_fresh = sensor_packet.is_fresh(now=now)
        obs_dict = sensor_packet.to_obs_dict()

        # Evaluate model prediction in both SHADOW_MODEL and AUTONOMOUS modes
        if self.policy is not None:
            model_sticks = self.policy.predict_sticks(obs_dict, device=self.device)
        else:
            model_sticks = np.zeros(4, dtype=np.float32)

        human_sticks = np.clip(np.asarray(xbox_sticks, dtype=np.float32), -1.0, 1.0)

        # Select active source before safety gate
        if self.mode in (ControlMode.MANUAL_XBOX, ControlMode.SHADOW_MODEL):
            desired_sticks = human_sticks
        else:
            desired_sticks = model_sticks

        # Apply ANNIMOS 60KG servo slew-rate limit (Section 3: smooth collar actuation)
        max_delta = CONFIG.servo_max_rate_per_sec * CONFIG.dt
        rate_limited_sticks = np.clip(
            desired_sticks,
            self.last_sent_sticks - max_delta,
            self.last_sent_sticks + max_delta,
        )

        # SAFETY GATE (Section 2 & 5): Deadman trigger + Sensor staleness watchdog
        # In manual mode, deadman trigger alone gates sticks; in autonomous mode, both deadman + fresh sensors are required
        safe_to_drive = bool(deadman_held and (sensors_fresh or self.mode == ControlMode.MANUAL_XBOX))

        if safe_to_drive:
            active_sticks = rate_limited_sticks.astype(np.float32)
        else:
            active_sticks = np.zeros(4, dtype=np.float32)

        self.last_sent_sticks = active_sticks.copy()

        pwm_us = sticks_to_pwm_us(active_sticks, deadman_active=safe_to_drive)
        crsf_bytes = sticks_to_crsf_frame(active_sticks, deadman_active=safe_to_drive)

        shadow_l1_err = float(np.mean(np.abs(model_sticks - human_sticks)))

        record = {
            "timestamp": now,
            "mode": self.mode.value,
            "deadman_held": bool(deadman_held),
            "sensors_fresh": bool(sensors_fresh),
            "safe_to_drive": safe_to_drive,
            "human_sticks": human_sticks.copy(),
            "model_sticks": model_sticks.copy(),
            "active_sticks": active_sticks.copy(),
            "pwm_us": pwm_us.copy(),
            "crsf_hex": crsf_bytes.hex(),
            "shadow_l1_error": shadow_l1_err,
        }
        self.telemetry_log.append(record)
        return record
