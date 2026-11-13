"""
One Core Simulator: JCB 8026 CTS 3D MuJoCo Environment (`sim/jcb8026_env.py`)
=============================================================================
Consolidates the repository around a single canonical 3D MuJoCo physics environment
engineered backwards from the physical JCB 8026 CTS deployment ("Two-week demo: hardware list"):

  - Actuators (50 Hz ELRS Servo Joystick Pattern — Section 3.3):
      Action `a_t in [-1, +1]^4` drives the 4 pilot joysticks via simulated
      ANNIMOS 60KG DS5160SSG servo dynamics (slew-rate limit + 4% pilot valve deadband):
        * `a[0]` -> **Channel 1: Left Fore-Aft (Dipper / Stick)**
        * `a[1]` -> **Channel 2: Left Side-to-Side (Slew / Swing)**
        * `a[2]` -> **Channel 3: Right Fore-Aft (Boom)**
        * `a[3]` -> **Channel 4: Right Side-to-Side (Bucket Curl)**

  - Sensors (Section 4):
      * `proprio` (`float32[12]`): 4x WitMotion WT901WIFI angles `[dipper, slew, boom, bucket]`
        + 4x angular velocities + 4x current servo stick positions.
      * `task_vec` (`float32[8]`): Overhead DJI Mini 4 Pro AprilTag relative target pose
        `[trench_dx, trench_dy, trench_dz, truck_dx, truck_dy, truck_dz, normalized_time, bucket_fill]`.
      * `livox_elevation` (`float32[1, 32, 32]`): Local 2.5D heightmap projected from the
        upside-down cab-roof Livox Mid360S 3D LiDAR point cloud.
      * `cab_rgb` (`float32[3, 64, 64]`): Cab-roof Reolink Duo 2 PoE camera crop along the boom
        (uses OpenGL `mujoco.Renderer` when requested, or fast geometric projection during headless batch RL).

  - Standardized Benchmark Scenarios (`IronBench-8026` T1..T5):
      * `T1_Nominal`:         Standard gravel trench + nominal dump truck pose
      * `T2_DeepTrench`:      Recessed gravel pile requiring deeper boom penetration
      * `T3_RandomTruckPose`: Randomized dump truck position & offset (tests AprilTag conditioning)
      * `T4_HeavyWetSoil`:    Higher gravel mass & friction (tests hydraulic crowd/curl torque)
      * `T5_Sim2Real_Stress`: Injects 40 ms Wi-Fi latency, WitMotion IMU noise, & 6% valve deadband
"""

import math
from typing import Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from deploy.jcb8026_config import CONFIG
from deploy.sensors_io import JCB8026SensorPacket, pointcloud_to_livox_heightmap


BENCHMARK_SCENARIOS = (
    "T1_Nominal",
    "T2_DeepTrench",
    "T3_RandomTruckPose",
    "T4_HeavyWetSoil",
    "T5_Sim2Real_Stress",
)


def build_jcb8026_mjcf(
    num_particles: int = 45,
    scenario: str = "T1_Nominal",
    seed: int = 42,
) -> Tuple[str, Dict[str, float]]:
    """
    Generates the MuJoCo XML (MJCF) for the JCB 8026 CTS mini-excavator,
    AprilTag markers (F1/F2), cab-roof Livox Mid360S & Reolink Duo 2 PoE mounts (C1/C3),
    dump truck, overhead DJI Mini 4 Pro drone (A1), and physical gravel particles.
    """
    rng = np.random.default_rng(seed)

    # Scenario-specific parameters
    trench_z_offset = -0.05 if scenario == "T2_DeepTrench" else 0.0
    truck_dx = float(rng.uniform(-0.22, 0.22)) if scenario == "T3_RandomTruckPose" else 0.0
    truck_dy = float(rng.uniform(-0.18, 0.18)) if scenario == "T3_RandomTruckPose" else 0.0
    particle_mass = 0.115 if scenario == "T4_HeavyWetSoil" else 0.08
    particle_friction = "1.85 0.07 0.03" if scenario == "T4_HeavyWetSoil" else "1.50 0.05 0.02"

    truck_x = 1.45 + truck_dx
    truck_y = 2.15 + truck_dy

    particle_xml = []
    for i in range(num_particles):
        layer = i // 15
        rem = i % 15
        row = rem // 5
        col = rem % 5
        px = 2.64 + col * 0.115 + rng.uniform(-0.012, 0.012)
        py = -0.14 + row * 0.14 + rng.uniform(-0.012, 0.012)
        pz = max(0.052, 0.06 + trench_z_offset + layer * 0.095)
        rad = rng.uniform(0.048, 0.056)
        r_c = rng.uniform(0.55, 0.74)
        g_c = rng.uniform(0.38, 0.52)
        b_c = rng.uniform(0.20, 0.32)
        particle_xml.append(
            f'    <body name="soil_{i}" pos="{px:.3f} {py:.3f} {pz:.3f}">\n'
            f'      <freejoint name="soil_j_{i}"/>\n'
            f'      <geom name="soil_g_{i}" type="sphere" contype="3" conaffinity="3" size="{rad:.3f}" '
            f'mass="{particle_mass:.3f}" friction="{particle_friction}" rgba="{r_c:.2f} {g_c:.2f} {b_c:.2f} 1"/>\n'
            f'    </body>'
        )
    particles_str = "\n".join(particle_xml)

    xml = f"""
<mujoco model="jcb_8026_cts">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default><geom density="250"/></default>
  <option timestep="0.004" gravity="0 0 -9.81" integrator="implicitfast"/>

  <visual>
    <headlight ambient="0.36 0.36 0.40" diffuse="0.68 0.68 0.68" specular="0.22 0.22 0.22"/>
    <rgba haze="0.12 0.16 0.22 1"/>
    <global offwidth="1280" offheight="720" elevation="-22" azimuth="135"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.16 0.24 0.36" rgb2="0.06 0.09 0.14" width="512" height="512"/>
    <texture name="ground_tex" type="2d" builtin="checker" rgb1="0.26 0.22 0.18" rgb2="0.22 0.18 0.15" width="512" height="512" mark="edge" markrgb="0.30 0.26 0.21"/>
    <material name="ground_mat" texture="ground_tex" texrepeat="12 12" reflectance="0.05"/>
    <material name="jcb_yellow" rgba="0.96 0.72 0.06 1" specular="0.65" shininess="0.55"/>
    <material name="dark_steel" rgba="0.18 0.20 0.23 1" specular="0.4" shininess="0.3"/>
    <material name="hydraulic_chrome" rgba="0.78 0.82 0.86 1" specular="0.9" shininess="0.8"/>
    <material name="truck_orange" rgba="0.90 0.36 0.12 1" specular="0.5" shininess="0.4"/>
    <material name="cab_glass" rgba="0.35 0.65 0.85 0.42" specular="0.8" shininess="0.8"/>
    <material name="apriltag_white" rgba="0.96 0.96 0.96 1" specular="0.1" shininess="0.1"/>
    <material name="sensor_blue" rgba="0.12 0.55 0.95 1" specular="0.7" shininess="0.6"/>
  </asset>

  <worldbody>
    <light pos="3 -4 8" dir="-0.3 0.4 -0.85" diffuse="0.85 0.82 0.76" specular="0.3 0.3 0.3" castshadow="true"/>
    <light pos="-4 4 6" dir="0.4 -0.4 -0.8" diffuse="0.35 0.40 0.48" castshadow="false"/>

    <!-- External Cinema Camera -->
    <camera name="cinema_cam" pos="5.2 -4.0 3.2" xyaxes="0.62 0.78 0 -0.36 0.28 0.89" fovy="46"/>

    <!-- Ground Plane -->
    <geom name="ground" type="plane" size="12 12 0.1" material="ground_mat" friction="1.0 0.02 0.01"/>

    <!-- Dig Trench Containment Walls (contype=2 so gravel stays piled without snagging bucket) -->
    <geom name="trench_back" type="box" pos="2.48 0.0 0.24" size="0.06 0.42 0.16" contype="2" conaffinity="2" rgba="0.36 0.28 0.20 1" friction="1.0 0.02 0.01"/>
    <geom name="trench_left" type="box" pos="2.88 0.28 0.22" size="0.55 0.05 0.14" contype="2" conaffinity="2" rgba="0.36 0.28 0.20 1"/>
    <geom name="trench_right" type="box" pos="2.88 -0.28 0.22" size="0.55 0.05 0.14" contype="2" conaffinity="2" rgba="0.36 0.28 0.20 1"/>
    <site name="trench_center" pos="2.85 0.0 0.12" size="0.05" rgba="1 0.8 0 0.5"/>

    <!-- ================= 1. JCB 8026 CTS MINI-EXCAVATOR (Origin 0, 0, 0) ================= -->
    <body name="jcb8026_undercarriage" pos="0 0 0">
      <geom name="track_left" type="box" pos="0 0.65 0.22" size="1.05 0.22 0.22" material="dark_steel"/>
      <geom name="track_right" type="box" pos="0 -0.65 0.22" size="1.05 0.22 0.22" material="dark_steel"/>
      <geom name="undercarriage_bridge" type="cylinder" pos="0 0 0.32" size="0.55 0.15" material="dark_steel"/>

      <!-- Rotating Cab (ISO Ch 2: Slew / Swing) -->
      <body name="jcb8026_cab" pos="0 0 0.52" gravcomp="1">
        <joint name="slew_joint" type="hinge" axis="0 0 1" range="-2.2 2.2" damping="250.0" armature="5.0"/>
        <geom name="cab_deck" type="box" pos="-0.35 0 0.25" size="0.75 0.62 0.26" material="jcb_yellow"/>
        <geom name="counterweight" type="box" pos="-0.98 0 0.28" size="0.22 0.60 0.30" material="dark_steel"/>
        <geom name="operator_cabin" type="box" pos="0.05 0.34 0.68" size="0.45 0.26 0.38" material="cab_glass"/>
        <geom name="cabin_roof" type="box" pos="0.05 0.34 1.08" size="0.47 0.28 0.04" material="jcb_yellow"/>

        <!-- F1/F2: Matte-Laminated A3 AprilTag on Cab Roof -->
        <geom name="apriltag_cab" type="box" pos="0.02 0.34 1.125" size="0.16 0.16 0.005" material="apriltag_white" contype="0" conaffinity="0"/>
        <!-- C1: Upside-down Livox Mid360S LiDAR under magnetic front roof plate + C3: Reolink Duo 2 PoE -->
        <geom name="livox_mid360s" type="cylinder" pos="0.48 0.34 1.03" size="0.055 0.04" material="sensor_blue" contype="0" conaffinity="0"/>
        <geom name="reolink_duo2_cam" type="box" pos="0.46 0.18 1.13" size="0.05 0.08 0.04" material="sensor_blue" contype="0" conaffinity="0"/>

        <!-- C3: Reolink Cab Roof Camera looking along the boom -->
        <camera name="cab_cam" pos="-0.10 0.48 1.15" xyaxes="0.18 -0.98 0 0.36 0.07 0.93" fovy="58"/>

        <!-- Main Boom Link (ISO Ch 3: Right Fore-Aft Boom) -->
        <body name="jcb8026_boom" pos="0.35 0.0 0.38" gravcomp="1">
          <joint name="boom_joint" type="hinge" axis="0 -1 0" range="-0.85 0.95" damping="250.0" armature="5.0"/>
          <geom name="boom_lower" type="capsule" fromto="0 0 0 0.95 0 0.68" size="0.12" material="jcb_yellow"/>
          <geom name="boom_upper" type="capsule" fromto="0.95 0 0.68 1.80 0 0.18" size="0.10" material="jcb_yellow"/>
          <geom name="boom_piston" type="cylinder" fromto="-0.15 0 -0.15 0.85 0 0.48" size="0.05" material="hydraulic_chrome"/>
          <!-- AprilTag + WitMotion WT901WIFI on Boom -->
          <geom name="apriltag_boom" type="box" pos="0.95 0.11 0.68" size="0.12 0.01 0.12" material="apriltag_white" contype="0" conaffinity="0"/>
          <geom name="wt901_boom" type="box" pos="0.65 0.11 0.48" size="0.03 0.02 0.03" material="sensor_blue" contype="0" conaffinity="0"/>

          <!-- Dipper / Stick Link (ISO Ch 1: Left Fore-Aft Dipper) -->
          <body name="jcb8026_dipper" pos="1.80 0 0.18" gravcomp="1">
            <joint name="dipper_joint" type="hinge" axis="0 -1 0" range="-1.35 1.15" damping="200.0" armature="4.0"/>
            <geom name="dipper_beam" type="capsule" fromto="0 0 0 1.15 0 -0.65" size="0.085" material="jcb_yellow"/>
            <geom name="dipper_piston" type="cylinder" fromto="-0.40 0 0.24 0.40 0 -0.12" size="0.04" material="hydraulic_chrome"/>
            <!-- AprilTag + WitMotion WT901WIFI on Dipper -->
            <geom name="apriltag_dipper" type="box" pos="0.55 0.09 -0.30" size="0.10 0.01 0.10" material="apriltag_white" contype="0" conaffinity="0"/>
            <geom name="wt901_dipper" type="box" pos="0.35 0.09 -0.18" size="0.03 0.02 0.03" material="sensor_blue" contype="0" conaffinity="0"/>

            <!-- 5-Wall Scoop Bucket (ISO Ch 4: Right Side Bucket) -->
            <body name="jcb8026_bucket" pos="1.15 0 -0.65" gravcomp="1">
              <joint name="bucket_joint" type="hinge" axis="0 -1 0" range="-1.65 1.45" damping="120.0" armature="2.5"/>
              <geom name="bkt_back" type="box" pos="0.0 0 -0.18" size="0.02 0.28 0.18" material="dark_steel" friction="1.5 0.05 0.02"/>
              <geom name="bkt_floor" type="box" pos="-0.22 0 -0.34" size="0.24 0.28 0.02" material="jcb_yellow" friction="1.5 0.05 0.02"/>
              <geom name="bkt_lip" type="box" pos="-0.44 0 -0.27" euler="0 0.55 0" size="0.08 0.28 0.02" material="dark_steel"/>
              <geom name="bkt_left" type="box" pos="-0.20 0.27 -0.18" size="0.22 0.02 0.16" material="jcb_yellow"/>
              <geom name="bkt_right" type="box" pos="-0.20 -0.27 -0.18" size="0.22 0.02 0.16" material="jcb_yellow"/>
              <geom name="wt901_bucket" type="box" pos="0.02 0.20 -0.05" size="0.03 0.02 0.03" material="sensor_blue" contype="0" conaffinity="0"/>
              <site name="bucket_tip" pos="-0.22 0 -0.20" size="0.04" rgba="0 1 0 1"/>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- ================= 2. DUMP TRUCK ================= -->
    <body name="dump_truck" pos="{truck_x:.3f} {truck_y:.3f} 0.0">
      <joint name="truck_drive" type="slide" axis="-1 0 0" range="0 4.5" damping="300.0" armature="5.0"/>
      <geom name="truck_chassis" type="box" pos="0 0 0.32" size="1.45 0.68 0.12" material="dark_steel"/>
      <geom name="wheel_fl" type="cylinder" pos="-0.95 0.72 0.28" euler="1.57 0 0" size="0.28 0.12" material="dark_steel"/>
      <geom name="wheel_fr" type="cylinder" pos="-0.95 -0.72 0.28" euler="1.57 0 0" size="0.28 0.12" material="dark_steel"/>
      <geom name="wheel_rl" type="cylinder" pos="0.75 0.72 0.28" euler="1.57 0 0" size="0.28 0.14" material="dark_steel"/>
      <geom name="wheel_rr" type="cylinder" pos="0.75 -0.72 0.28" euler="1.57 0 0" size="0.28 0.14" material="dark_steel"/>
      <geom name="truck_cab" type="box" pos="-1.05 0 0.78" size="0.42 0.62 0.36" material="cab_glass"/>
      <geom name="truck_hood" type="box" pos="-1.32 0 0.56" size="0.22 0.60 0.18" material="truck_orange"/>

      <!-- 5-Wall Open Cargo Dump Bed -->
      <geom name="bed_floor" type="box" pos="0.35 0 0.46" size="0.95 0.68 0.04" material="truck_orange"/>
      <geom name="bed_front_wall" type="box" pos="-0.58 0 0.78" size="0.04 0.68 0.32" material="truck_orange"/>
      <geom name="bed_back_wall" type="box" pos="1.28 0 0.74" size="0.04 0.68 0.28" material="truck_orange"/>
      <geom name="bed_left_wall" type="box" pos="0.35 0.66 0.74" size="0.95 0.04 0.28" material="truck_orange"/>
      <geom name="bed_right_wall" type="box" pos="0.35 -0.66 0.74" size="0.95 0.04 0.28" material="truck_orange"/>
      <site name="truck_bed_center" pos="0.35 0 0.65" size="0.06" rgba="0 0.8 1 1"/>
    </body>

    <!-- ================= 3. A1: DJI MINI 4 PRO SURVEY DRONE ================= -->
    <body name="dji_mini4_pro" pos="1.4 1.1 5.8">
      <geom name="drone_core" type="box" size="0.18 0.18 0.05" rgba="0.15 0.85 0.95 1" contype="0" conaffinity="0"/>
      <geom name="drone_arm1" type="capsule" fromto="-0.28 -0.28 0 0.28 0.28 0" size="0.02" rgba="0.9 0.9 0.9 1" contype="0" conaffinity="0"/>
      <geom name="drone_arm2" type="capsule" fromto="-0.28 0.28 0 0.28 -0.28 0" size="0.02" rgba="0.9 0.9 0.9 1" contype="0" conaffinity="0"/>
      <camera name="drone_cam" pos="0 0 -0.15" xyaxes="0 -1 0 1 0 0" fovy="58"/>
    </body>

    <!-- ================= 4. PHYSICAL GRAVEL PARTICLES ================= -->
{particles_str}
  </worldbody>

  <!-- Actuators ordered in exact ISO Pattern: Ch1 Dipper, Ch2 Slew, Ch3 Boom, Ch4 Bucket (+ Truck Drive) -->
  <actuator>
    <position name="ch1_dipper_act" joint="dipper_joint" kp="6500" ctrlrange="-1.35 1.15"/>
    <position name="ch2_slew_act"   joint="slew_joint"   kp="6000" ctrlrange="-2.0 2.0"/>
    <position name="ch3_boom_act"   joint="boom_joint"   kp="8000" ctrlrange="-0.85 0.95"/>
    <position name="ch4_bucket_act" joint="bucket_joint" kp="4500" ctrlrange="-1.65 1.45"/>
    <position name="truck_act"      joint="truck_drive"  kp="3500" ctrlrange="0 4.5"/>
  </actuator>
</mujoco>
"""
    meta = {
        "truck_x": truck_x,
        "truck_y": truck_y,
        "particle_mass": particle_mass,
        "num_particles": float(num_particles),
    }
    return xml, meta


class JCB8026BenchEnv(gym.Env):
    """
    Canonical JCB 8026 CTS Gymnasium Environment (`JCB8026BenchEnv-v0`).
    Matches the 50 Hz ELRS 4-channel ISO joystick interface and sensor suite.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        scenario: str = "T1_Nominal",
        num_particles: int = 45,
        control_hz: int = 25,  # 25 Hz sim step (10 physics substeps @ 4ms) or 50 Hz (5 substeps)
        horizon_sec: float = 10.5,
        use_opengl_cam: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        assert scenario in BENCHMARK_SCENARIOS, f"Unknown scenario {scenario}"
        self.scenario = scenario
        self.num_particles = num_particles
        self.control_hz = control_hz
        self.dt = 1.0 / float(control_hz)
        self.horizon_sec = horizon_sec
        self.max_steps = int(round(horizon_sec * control_hz))
        self.use_opengl_cam = use_opengl_cam
        self.base_seed = seed

        xml_str, self.scene_meta = build_jcb8026_mjcf(
            num_particles=num_particles, scenario=scenario, seed=seed
        )
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)
        self.substeps = max(1, int(round(self.dt / self.model.opt.timestep)))

        # Cache MuJoCo IDs in exact ISO Channel order: [Ch1 Dipper, Ch2 Slew, Ch3 Boom, Ch4 Bucket]
        self.iso_joint_names = ["dipper_joint", "slew_joint", "boom_joint", "bucket_joint"]
        self.iso_qpos_idx = np.array(
            [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.iso_joint_names],
            dtype=np.int32,
        )
        self.iso_qvel_idx = np.array(
            [self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.iso_joint_names],
            dtype=np.int32,
        )
        self.soil_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"soil_{i}")
            for i in range(num_particles)
        ]
        self.bucket_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "bucket_tip")
        self.truck_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "truck_bed_center")
        self.trench_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "trench_center")

        # Gym Spaces matching `models/policy.py` and `deploy/jcb8026_config.py`
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "proprio": spaces.Box(-np.inf, np.inf, shape=(CONFIG.proprio_dim,), dtype=np.float32),
                "task_vec": spaces.Box(-np.inf, np.inf, shape=(CONFIG.task_vec_dim,), dtype=np.float32),
                "livox_elevation": spaces.Box(-np.inf, np.inf, shape=CONFIG.elevation_shape, dtype=np.float32),
                "cab_rgb": spaces.Box(0.0, 1.0, shape=CONFIG.cab_rgb_shape, dtype=np.float32),
            }
        )

        self._small_renderer = None
        self.rng = np.random.default_rng(seed)

        # Internal state for servo lag, total variation, and cycle tracking
        self.step_count = 0
        self.servo_sticks = np.zeros(4, dtype=np.float32)
        self.prev_sticks = np.zeros(4, dtype=np.float32)
        self.hydraulic_target_rad = np.zeros(4, dtype=np.float32)
        self.peak_bucket_count = 0
        self.total_variation_sum = 0.0
        self._obs_delay_buffer = []

    def get_iso_qpos(self) -> np.ndarray:
        """Returns joint angles in ISO order: [Ch1 Dipper, Ch2 Slew, Ch3 Boom, Ch4 Bucket]."""
        return self.data.qpos[self.iso_qpos_idx].astype(np.float32).copy()

    def get_iso_qvel(self) -> np.ndarray:
        """Returns joint angular velocities in ISO order: [Ch1 Dipper, Ch2 Slew, Ch3 Boom, Ch4 Bucket]."""
        return self.data.qvel[self.iso_qvel_idx].astype(np.float32).copy()

    def expert_iso_sticks(self, t_sec: Optional[float] = None) -> Tuple[np.ndarray, str]:
        """
        Scripted expert controller in the 4-channel ISO joystick space `[-1, +1]^4`
        (`[Ch1 Dipper, Ch2 Slew, Ch3 Boom, Ch4 Bucket]`), conditioned on the overhead
        drone's AprilTag truck pose so it works across all scenarios T1..T5.
        """
        if t_sec is None:
            t_sec = self.step_count * self.dt

        # Compute target slew angle toward the current truck bed center via AprilTag pose
        bed_pos = self.data.site_xpos[self.truck_site_id]
        target_slew = float(np.clip(math.atan2(bed_pos[1], bed_pos[0]) - 0.02, -1.35, 1.35))
        deep_offset = -0.04 if self.scenario == "T2_DeepTrench" else 0.0

        # Waypoints in ISO joint space: [Ch1 Dipper, Ch2 Slew, Ch3 Boom, Ch4 Bucket]
        waypoints = [
            (0.0,  np.array([ 0.55, 0.00,         0.35,               -0.40], dtype=np.float32), "1. ALIGN_TRENCH"),
            (1.2,  np.array([ 0.55, 0.00,        -0.18 + deep_offset,  0.15], dtype=np.float32), "2. PENETRATE_SOIL"),
            (2.8,  np.array([-0.22, 0.00,        -0.02 + deep_offset,  0.35], dtype=np.float32), "3. RAKE_AND_SCOOP"),
            (3.8,  np.array([-0.32, 0.00,         0.08,               -0.35], dtype=np.float32), "4. CURL_AND_TRAP"),
            (5.0,  np.array([-0.30, 0.00,         0.60,               -0.55], dtype=np.float32), "5. LIFT_BUCKET"),
            (7.0,  np.array([-0.30, target_slew,  0.60,               -0.55], dtype=np.float32), "6. SLEW_TO_TRUCK"),
            (9.2,  np.array([-0.30, target_slew,  0.50,                0.72], dtype=np.float32), "7. DUMP_INTO_BED"),
            (10.5, np.array([-0.35, target_slew * 0.6, 0.56,          -0.42], dtype=np.float32), "8. RETURN_SWING"),
        ]

        q_des = waypoints[-1][1].copy()
        phase_name = waypoints[-1][2]
        for i in range(len(waypoints) - 1):
            t0, q0, p0 = waypoints[i]
            t1, q1, p1 = waypoints[i + 1]
            if t0 <= t_sec <= t1:
                alpha = (t_sec - t0) / max(1e-5, (t1 - t0))
                s = 0.5 * (1.0 - math.cos(math.pi * alpha))
                q_des = ((1.0 - s) * q0 + s * q1).astype(np.float32)
                phase_name = p1
                break

        # Convert desired joint target into normalized ISO stick command [-1, +1]^4
        q_cur = self.get_iso_qpos()
        max_step_rad = np.asarray(CONFIG.max_joint_vel_rad_s, dtype=np.float32) * (self.dt * 4.5)
        sticks = np.clip((q_des - q_cur) / max_step_rad, -1.0, 1.0)
        return sticks.astype(np.float32), phase_name

    def count_particles(self) -> Dict[str, int]:
        soil_pos = self.data.xpos[self.soil_body_ids]
        bkt_pos = self.data.site_xpos[self.bucket_site_id]
        bed_pos = self.data.site_xpos[self.truck_site_id]

        d_bkt = np.linalg.norm(soil_pos - bkt_pos, axis=1)
        in_bucket = int(np.sum((d_bkt < 0.36) & (soil_pos[:, 2] > 0.25)))

        in_truck = int(
            np.sum(
                (np.abs(soil_pos[:, 0] - bed_pos[0]) < 0.90)
                & (np.abs(soil_pos[:, 1] - bed_pos[1]) < 0.65)
                & (soil_pos[:, 2] > 0.42)
            )
        )

        # Spilled = lifted out of the trench (x < 2.35 or y > 0.45) but resting on ground outside truck
        spilled = int(
            np.sum(
                ((soil_pos[:, 0] < 2.35) | (soil_pos[:, 1] > 0.45))
                & (soil_pos[:, 2] < 0.35)
            )
        )
        return {"in_bucket": in_bucket, "in_truck": in_truck, "spilled": spilled}

    def _build_sensor_packet(self) -> JCB8026SensorPacket:
        qpos = self.get_iso_qpos()
        qvel = self.get_iso_qvel()

        if self.scenario == "T5_Sim2Real_Stress":
            qpos = qpos + self.rng.normal(0.0, 0.015, size=4).astype(np.float32)
            qvel = qvel + self.rng.normal(0.0, 0.03, size=4).astype(np.float32)

        proprio = np.concatenate([qpos, qvel, self.servo_sticks], axis=0).astype(np.float32)

        bkt_pos = self.data.site_xpos[self.bucket_site_id].astype(np.float32)
        trench_pos = self.data.site_xpos[self.trench_site_id].astype(np.float32)
        bed_pos = self.data.site_xpos[self.truck_site_id].astype(np.float32)
        counts = self.count_particles()

        norm_time = float(self.step_count) / max(1.0, float(self.max_steps))
        bucket_fill = float(counts["in_bucket"]) / 18.0  # 18 particles = 100% rated bucket fill

        task_vec = np.array(
            [
                trench_pos[0] - bkt_pos[0],
                trench_pos[1] - bkt_pos[1],
                trench_pos[2] - bkt_pos[2],
                bed_pos[0] - bkt_pos[0],
                bed_pos[1] - bkt_pos[1],
                bed_pos[2] - bkt_pos[2],
                norm_time,
                bucket_fill,
            ],
            dtype=np.float32,
        )

        # C1: Livox Mid360S 1x32x32 local elevation map from 3D soil + bucket + truck points
        soil_xyz = self.data.xpos[self.soil_body_ids].copy()
        livox_map = pointcloud_to_livox_heightmap(soil_xyz, grid_size=32)

        # C3: Reolink Duo 2 PoE 3x64x64 cab-roof view
        if self.use_opengl_cam:
            if self._small_renderer is None:
                self._small_renderer = mujoco.Renderer(self.model, height=64, width=64)
            self._small_renderer.update_scene(self.data, camera="cab_cam")
            rgb_u8 = self._small_renderer.render()
            cab_rgb = (rgb_u8.astype(np.float32) / 255.0).transpose(2, 0, 1)
        else:
            # Fast deterministic 3-channel feature projection when running headless CPU/MPS batch RL
            cab_rgb = np.zeros((3, 64, 64), dtype=np.float32)
            # Channel 0: upsampled terrain elevation, Channel 1: bucket spatial mask, Channel 2: truck bed mask
            cab_rgb[0] = np.repeat(np.repeat((livox_map[0] + 0.35), 2, axis=0), 2, axis=1)
            bx = int(np.clip((bkt_pos[0] - 0.5) / 3.2 * 63, 2, 61))
            by = int(np.clip((bkt_pos[1] + 2.2) / 3.2 * 63, 2, 61))
            cab_rgb[1, bx - 2 : bx + 3, by - 2 : by + 3] = 1.0
            tx = int(np.clip((bed_pos[0] - 0.5) / 3.2 * 63, 3, 60))
            ty = int(np.clip((bed_pos[1] + 2.2) / 3.2 * 63, 3, 60))
            cab_rgb[2, tx - 3 : tx + 4, ty - 3 : ty + 4] = 1.0

        now = self.step_count * self.dt
        return JCB8026SensorPacket(
            timestamp=now,
            proprio=proprio,
            task_vec=task_vec,
            livox_elevation=livox_map,
            cab_rgb=cab_rgb,
            t_wt901_udp=now,
            t_livox_lidar=now,
            t_reolink_rtsp=now,
            t_drone_apriltag=now,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            xml_str, self.scene_meta = build_jcb8026_mjcf(
                num_particles=self.num_particles, scenario=self.scenario, seed=seed
            )
            self.model = mujoco.MjModel.from_xml_string(xml_str)
            self.data = mujoco.MjData(self.model)
            if self._small_renderer is not None:
                self._small_renderer = mujoco.Renderer(self.model, height=64, width=64)

        mujoco.mj_resetData(self.model, self.data)

        # Initial JCB 8026 pose over trench in ISO order: [Dipper=0.55, Slew=0.0, Boom=0.35, Bucket=-0.40]
        init_iso = np.array([0.55, 0.00, 0.35, -0.40], dtype=np.float32)
        init_iso += self.rng.uniform(-0.015, 0.015, size=4).astype(np.float32)

        for idx, q_val in zip(self.iso_qpos_idx, init_iso):
            self.data.qpos[idx] = float(q_val)
        self.hydraulic_target_rad = init_iso.copy()
        self.data.ctrl[:4] = init_iso.copy()
        self.data.ctrl[4] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.servo_sticks = np.zeros(4, dtype=np.float32)
        self.prev_sticks = np.zeros(4, dtype=np.float32)
        self.peak_bucket_count = 0
        self.total_variation_sum = 0.0
        self._obs_delay_buffer = []

        pkt = self._build_sensor_packet()
        obs = pkt.to_obs_dict()
        self._obs_delay_buffer.append(obs)
        return obs, {"scenario": self.scenario, "counts": self.count_particles()}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        raw_sticks = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # 1. ANNIMOS 60KG DS5160SSG servo slew-rate limit & pilot valve deadband (Section 3)
        max_delta = CONFIG.servo_max_rate_per_sec * self.dt
        self.servo_sticks = np.clip(
            raw_sticks,
            self.servo_sticks - max_delta,
            self.servo_sticks + max_delta,
        )
        deadband = 0.06 if self.scenario == "T5_Sim2Real_Stress" else CONFIG.joystick_deadband
        effective_sticks = np.where(np.abs(self.servo_sticks) < deadband, 0.0, self.servo_sticks)

        # Track servo smoothness (Total Variation)
        step_tv = float(np.mean(np.abs(self.servo_sticks - self.prev_sticks)))
        self.total_variation_sum += step_tv
        self.prev_sticks = self.servo_sticks.copy()

        # 2. Integrate pilot stick deflection into hydraulic joint target (ISO Ch 1..4)
        max_vel = np.asarray(CONFIG.max_joint_vel_rad_s, dtype=np.float32)
        q_cur = self.get_iso_qpos()
        self.hydraulic_target_rad = q_cur + effective_sticks * max_vel * (self.dt * 4.5)

        # Clamp to JCB 8026 actuator limits
        ctrl_low = np.array([-1.35, -2.0, -0.85, -1.65], dtype=np.float32)
        ctrl_high = np.array([1.15, 2.0, 0.95, 1.45], dtype=np.float32)
        self.hydraulic_target_rad = np.clip(self.hydraulic_target_rad, ctrl_low, ctrl_high)

        self.data.ctrl[:4] = self.hydraulic_target_rad
        self.data.ctrl[4] = 0.0

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        counts = self.count_particles()
        self.peak_bucket_count = max(self.peak_bucket_count, counts["in_bucket"])

        # 3. Dense Excavation & Truck-Loading Reward + Smoothness Regularization
        expert_sticks, phase_name = self.expert_iso_sticks()
        imitation_bonus = float(np.exp(-2.5 * np.mean(np.square(raw_sticks - expert_sticks))))
        reward = (
            1.5 * imitation_bonus
            + 0.25 * float(counts["in_bucket"])
            + 1.20 * float(counts["in_truck"])
            - 0.15 * float(counts["spilled"])
            - 0.35 * step_tv
        )

        terminated = False
        truncated = bool(self.step_count >= self.max_steps)

        pkt = self._build_sensor_packet()
        obs = pkt.to_obs_dict()

        # Optional 1-step (40 ms) Wi-Fi latency buffer in T5_Sim2Real_Stress
        if self.scenario == "T5_Sim2Real_Stress" and len(self._obs_delay_buffer) > 0:
            delayed_obs = self._obs_delay_buffer[-1]
            self._obs_delay_buffer.append(obs)
            obs_out = delayed_obs
        else:
            obs_out = obs

        info = {
            "scenario": self.scenario,
            "phase": phase_name,
            "in_bucket": counts["in_bucket"],
            "in_truck": counts["in_truck"],
            "spilled": counts["spilled"],
            "peak_bucket": self.peak_bucket_count,
            "step_tv": step_tv,
            "mean_tv": self.total_variation_sum / max(1, self.step_count),
            "particle_mass_kg": self.scene_meta["particle_mass"],
            "truck_mass_kg": counts["in_truck"] * self.scene_meta["particle_mass"],
        }
        return obs_out, float(reward), terminated, truncated, info
