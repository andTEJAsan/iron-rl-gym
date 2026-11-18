"""
JCB 8026 Cab-Only Sensor Suite I/O & Staleness Watchdog (v3 — Drone Dropped)
============================================================================
Handles the 3 physical cab-mounted sensor streams arriving at the ground laptop:
  1. C4: 4x WitMotion WT901WIFI Wireless Tilt Sensors (Boom, Dipper, Bucket link, Cab yaw)
         in Station mode -> UDP packets over GL.iNet Opal (C5) Wi-Fi.
  2. C1: Livox Mid360S 3D LiDAR mounted upside-down under cab roof front plate, tilted
         toward bucket -> RJ45 192.168.1.x -> projected into a 1x32x32 local heightmap.
  3. C3: Reolink Duo 2 PoE 4K Dual-Lens 180° Camera on cab roof looking along the boom
         and across the slew arc -> RTSP stream -> resized to 3x64x64 RGB tensor.

Because the DJI Mini 4 Pro drone was dropped for the MVP (Collaborator Note #1),
`task_vec` (8D) is now computed 100% from the cab sensors (`C1` Livox heightmap +
`C3` Reolink 180° truck AprilTag/bbox detection + `C4` WitMotion cab slew yaw IMU),
with optional semantic overrides from `deploy/vlm_supervisor.py`.
"""

from dataclasses import dataclass
import math
import time
from typing import Dict, Tuple
import numpy as np
from deploy.jcb8026_config import CONFIG


@dataclass
class JCB8026SensorPacket:
    timestamp: float
    # C4: 4x WitMotion WT901WIFI angles [dipper, slew, boom, bucket] (rad) + velocities (rad/s) + prev_action
    proprio: np.ndarray          # float32[12]
    # Cab-derived target vector [dig_dx, dig_dy, dig_dz, truck_dx, truck_dy, truck_dz, phase_prog, bucket_fill]
    task_vec: np.ndarray         # float32[8]
    # C1: Livox Mid360S local 2.5D elevation heightmap (m) around trench & truck
    livox_elevation: np.ndarray  # float32[1, 32, 32]
    # C3: Reolink Duo 2 PoE cab-roof RGB image crop [0..1]
    cab_rgb: np.ndarray          # float32[3, 64, 64]

    # Individual stream timestamps for the 200 ms staleness watchdog (No drone required!)
    t_wt901_udp: float = 0.0
    t_livox_lidar: float = 0.0
    t_reolink_rtsp: float = 0.0

    def is_fresh(self, now: float = None, max_age_sec: float = CONFIG.sensor_staleness_timeout_sec) -> bool:
        """
        Hardware Spec Section 2: 'Always: the program sends neutral unless the deadman
        trigger is held, and if the video, lidar or arm angles go stale.'
        """
        if now is None:
            now = time.monotonic()
        ages = [
            now - self.t_wt901_udp,
            now - self.t_livox_lidar,
            now - self.t_reolink_rtsp,
        ]
        return all(0.0 <= age <= max_age_sec for age in ages)

    def to_obs_dict(self) -> Dict[str, np.ndarray]:
        return {
            "proprio": self.proprio.astype(np.float32),
            "task_vec": self.task_vec.astype(np.float32),
            "livox_elevation": self.livox_elevation.astype(np.float32),
            "cab_rgb": self.cab_rgb.astype(np.float32),
        }


def compute_cab_ego_task_vec(
    bucket_tip_xyz: np.ndarray,
    cab_slew_yaw_rad: float,
    dig_target_odom_xyz: np.ndarray,
    truck_bed_odom_xyz: np.ndarray,
    norm_time: float,
    bucket_fill_ratio: float,
) -> np.ndarray:
    """
    Replaces the dropped overhead drone with Cab-Ego Target Estimation:
    Uses the cab-roof Livox Mid360S (C1) + Reolink 180° Duo 2 (C3) + WitMotion Cab IMU (C4)
    to track the relative 3D offset from the bucket tip to the optimal soil peak and dump truck bed.
    """
    # Relative vectors in odometry / base frame (tracked via cab slew yaw angle)
    d_dig = dig_target_odom_xyz - bucket_tip_xyz
    d_truck = truck_bed_odom_xyz - bucket_tip_xyz

    return np.array(
        [
            float(d_dig[0]),
            float(d_dig[1]),
            float(d_dig[2]),
            float(d_truck[0]),
            float(d_truck[1]),
            float(d_truck[2]),
            float(norm_time),
            float(bucket_fill_ratio),
        ],
        dtype=np.float32,
    )


def pointcloud_to_livox_heightmap(
    points_xyz: np.ndarray,
    x_range: Tuple[float, float] = (0.5, 3.7),
    y_range: Tuple[float, float] = (-2.2, 1.0),
    grid_size: int = 32,
) -> np.ndarray:
    """
    Projects 3D points (from upside-down cab-roof Livox Mid360S or MuJoCo deformable soil terrain)
    into the canonical (1, 32, 32) elevation heightmap fed to the policy.
    """
    hmap = np.full((grid_size, grid_size), -0.35, dtype=np.float32)
    if points_xyz is None or len(points_xyz) == 0:
        return hmap[np.newaxis, :, :]

    xs = points_xyz[:, 0]
    ys = points_xyz[:, 1]
    zs = points_xyz[:, 2]

    ix = np.clip(((xs - x_range[0]) / (x_range[1] - x_range[0]) * grid_size).astype(np.int32), 0, grid_size - 1)
    iy = np.clip(((ys - y_range[0]) / (y_range[1] - y_range[0]) * grid_size).astype(np.int32), 0, grid_size - 1)

    np.maximum.at(hmap, (ix, iy), zs.astype(np.float32))
    return hmap[np.newaxis, :, :]
