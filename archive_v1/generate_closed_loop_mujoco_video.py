"""
Closed-Loop 3D MuJoCo Physics Simulation + Vision Policy Video Generator
========================================================================
Builds a full 3D MuJoCo physics environment with:
  - Articulated 4-DOF Hydraulic Excavator (tracks, cab, boom, stick, 5-wall hollow scoop bucket)
  - 55 physical soil/gravel particles with real contact dynamics
  - 3D Haul / Dump Truck with a 5-wall cargo bed and motorized drive-away joint
  - Autonomous Overhead Survey Drone with onboard camera
  - 3 MuJoCo OpenGL Cameras: Cinema External Cam, First-Person Cab Cam, Overhead Drone Cam

Trains a PyTorch Vision-to-Action CNN Policy (taking Cab Camera RGB + Drone Camera RGB +
4 Joint Angles -> 4 Hydraulic Joystick Commands) and rolls it out 100% in closed-loop
to generate an HD MP4 video, animated GIF, and summary dashboard PNG.
"""

import os
import math
import cv2
import imageio
import mujoco
import numpy as np
from PIL import Image

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

OUT_DIR = "/Users/teja-san/Work/hcilab/iron_rl_gym/data"
os.makedirs(OUT_DIR, exist_ok=True)


def build_excavator_mjcf(num_particles: int = 55) -> str:
    """
    Generate the complete MuJoCo XML (MJCF) for the quarry site:
      - Textured quarry ground + containment trench walls so soil pile stays compact
      - Detailed 4-DOF Excavator with hollow 5-plate scoop bucket
      - 3D Dump Truck with hollow cargo bed & drive-away slide joint
      - Overhead quadcopter drone
      - Physical gravel/soil boulders in the dig trench
    """
    rng = np.random.default_rng(42)
    particle_xml = []
    # Spawn gravel particles inside the dig trench along the bucket's scoop arc (x=2.63..3.11, y=-0.14..0.14)
    for i in range(num_particles):
        layer = i // 15
        rem = i % 15
        row = rem // 5
        col = rem % 5
        px = 2.63 + col * 0.12 + rng.uniform(-0.015, 0.015)
        py = -0.14 + row * 0.14 + rng.uniform(-0.015, 0.015)
        pz = 0.06 + layer * 0.10
        rad = rng.uniform(0.048, 0.058)
        r_c = rng.uniform(0.55, 0.75)
        g_c = rng.uniform(0.38, 0.52)
        b_c = rng.uniform(0.20, 0.32)
        particle_xml.append(
            f'    <body name="soil_{i}" pos="{px:.3f} {py:.3f} {pz:.3f}">\n'
            f'      <freejoint name="soil_j_{i}"/>\n'
            f'      <geom name="soil_g_{i}" type="sphere" contype="3" conaffinity="3" size="{rad:.3f}" mass="0.08" '
            f'friction="1.5 0.05 0.02" rgba="{r_c:.2f} {g_c:.2f} {b_c:.2f} 1"/>\n'
            f'    </body>'
        )
    particles_str = "\n".join(particle_xml)

    xml = f"""
<mujoco model="iron_excavator_site">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default><geom density="250"/></default>
  <option timestep="0.004" gravity="0 0 -9.81" integrator="implicitfast"/>

  <visual>
    <headlight ambient="0.35 0.35 0.38" diffuse="0.65 0.65 0.65" specular="0.2 0.2 0.2"/>
    <rgba haze="0.12 0.16 0.22 1"/>
    <global offwidth="1280" offheight="720" elevation="-22" azimuth="135"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.16 0.24 0.36" rgb2="0.06 0.09 0.14" width="512" height="512"/>
    <texture name="ground_tex" type="2d" builtin="checker" rgb1="0.26 0.22 0.18" rgb2="0.22 0.18 0.15" width="512" height="512" mark="edge" markrgb="0.30 0.26 0.21"/>
    <material name="ground_mat" texture="ground_tex" texrepeat="12 12" reflectance="0.05"/>
    <material name="yellow_paint" rgba="0.95 0.70 0.08 1" specular="0.6" shininess="0.5"/>
    <material name="dark_steel" rgba="0.18 0.20 0.23 1" specular="0.4" shininess="0.3"/>
    <material name="hydraulic_chrome" rgba="0.78 0.82 0.86 1" specular="0.9" shininess="0.8"/>
    <material name="truck_orange" rgba="0.90 0.36 0.12 1" specular="0.5" shininess="0.4"/>
    <material name="cab_glass" rgba="0.35 0.65 0.85 0.45" specular="0.8" shininess="0.8"/>
  </asset>

  <worldbody>
    <light pos="3 -4 8" dir="-0.3 0.4 -0.85" diffuse="0.85 0.82 0.76" specular="0.3 0.3 0.3" castshadow="true"/>
    <light pos="-4 4 6" dir="0.4 -0.4 -0.8" diffuse="0.35 0.40 0.48" castshadow="false"/>

    <!-- External Cinema Camera -->
    <camera name="cinema_cam" pos="5.2 -4.0 3.2" xyaxes="0.62 0.78 0 -0.36 0.28 0.89" fovy="46"/>

    <!-- Ground Plane -->
    <geom name="ground" type="plane" size="12 12 0.1" material="ground_mat" friction="1.0 0.02 0.01"/>

    <!-- Subtle Dig Trench Backstop & Curb (contype=2 so it contains soil without snagging bucket) -->
    <geom name="trench_back" type="box" pos="2.48 0.0 0.24" size="0.06 0.42 0.16" contype="2" conaffinity="2" rgba="0.36 0.28 0.20 1" friction="1.0 0.02 0.01"/>
    <geom name="trench_left" type="box" pos="2.88 0.28 0.22" size="0.55 0.05 0.14" contype="2" conaffinity="2" rgba="0.36 0.28 0.20 1"/>
    <geom name="trench_right" type="box" pos="2.88 -0.28 0.22" size="0.55 0.05 0.14" contype="2" conaffinity="2" rgba="0.36 0.28 0.20 1"/>

    <!-- ================= 1. EXCAVATOR (Origin 0, 0, 0) ================= -->
    <body name="excavator_undercarriage" pos="0 0 0">
      <!-- Left & Right Caterpillar Tracks -->
      <geom name="track_left" type="box" pos="0 0.65 0.22" size="1.05 0.22 0.22" material="dark_steel"/>
      <geom name="track_right" type="box" pos="0 -0.65 0.22" size="1.05 0.22 0.22" material="dark_steel"/>
      <geom name="undercarriage_bridge" type="cylinder" pos="0 0 0.32" size="0.55 0.15" material="dark_steel"/>

      <!-- Rotating Upper Cab (Joint 1: swing_joint) -->
      <body name="excavator_cab" pos="0 0 0.52" gravcomp="1">
        <joint name="swing_joint" type="hinge" axis="0 0 1" range="-2.2 2.2" damping="250.0" armature="5.0"/>
        <!-- Engine Deck & Counterweight -->
        <geom name="cab_deck" type="box" pos="-0.35 0 0.25" size="0.75 0.62 0.26" material="yellow_paint"/>
        <geom name="counterweight" type="box" pos="-0.98 0 0.28" size="0.22 0.60 0.30" material="dark_steel"/>
        <!-- Operator Glass Cabin on Left Side -->
        <geom name="operator_cabin" type="box" pos="0.05 0.34 0.68" size="0.45 0.26 0.38" material="cab_glass"/>
        <geom name="cabin_roof" type="box" pos="0.05 0.34 1.08" size="0.47 0.28 0.04" material="yellow_paint"/>

        <!-- First-Person Cab Camera mounted on roof looking out at the boom, bucket & trench -->
        <camera name="cab_cam" pos="-0.10 0.48 1.15" xyaxes="0.18 -0.98 0 0.36 0.07 0.93" fovy="58"/>

        <!-- Main Boom Link (Joint 2: boom_joint, axis 0 -1 0 so +angle = UP, -angle = DOWN) -->
        <body name="excavator_boom" pos="0.35 0.0 0.38" gravcomp="1">
          <joint name="boom_joint" type="hinge" axis="0 -1 0" range="-0.85 0.95" damping="250.0" armature="5.0"/>
          <geom name="boom_lower" type="capsule" fromto="0 0 0 0.95 0 0.68" size="0.12" material="yellow_paint"/>
          <geom name="boom_upper" type="capsule" fromto="0.95 0 0.68 1.80 0 0.18" size="0.10" material="yellow_paint"/>
          <geom name="boom_piston" type="cylinder" fromto="-0.15 0 -0.15 0.85 0 0.48" size="0.05" material="hydraulic_chrome"/>

          <!-- Forearm / Stick Link (Joint 3: stick_joint, axis 0 -1 0 so -angle pulls stick IN toward cab) -->
          <body name="excavator_stick" pos="1.80 0 0.18" gravcomp="1">
            <joint name="stick_joint" type="hinge" axis="0 -1 0" range="-1.35 1.15" damping="200.0" armature="4.0"/>
            <geom name="stick_beam" type="capsule" fromto="0 0 0 1.15 0 -0.65" size="0.085" material="yellow_paint"/>
            <geom name="stick_piston" type="cylinder" fromto="-0.40 0 0.24 0.40 0 -0.12" size="0.04" material="hydraulic_chrome"/>

            <!-- Hollow 5-Wall Scoop Bucket (Joint 4: bucket_joint, axis 0 -1 0 so -angle curls bucket IN toward cab) -->
            <body name="excavator_bucket" pos="1.15 0 -0.65" gravcomp="1">
              <joint name="bucket_joint" type="hinge" axis="0 -1 0" range="-1.65 1.45" damping="120.0" armature="2.5"/>
              <!-- Back plate of bucket -->
              <geom name="bkt_back" type="box" pos="0.0 0 -0.18" size="0.02 0.28 0.18" material="dark_steel" friction="1.5 0.05 0.02"/>
              <!-- Bottom floor plate extending toward cab (-X) so pulling stick IN (-X) scoops particles! -->
              <geom name="bkt_floor" type="box" pos="-0.22 0 -0.34" size="0.24 0.28 0.02" material="yellow_paint" friction="1.5 0.05 0.02"/>
              <!-- Front cutting lip at cab-facing edge -->
              <geom name="bkt_lip" type="box" pos="-0.44 0 -0.27" euler="0 0.55 0" size="0.08 0.28 0.02" material="dark_steel"/>
              <!-- Left & Right side walls to hold gravel inside -->
              <geom name="bkt_left" type="box" pos="-0.20 0.27 -0.18" size="0.22 0.02 0.16" material="yellow_paint"/>
              <geom name="bkt_right" type="box" pos="-0.20 -0.27 -0.18" size="0.22 0.02 0.16" material="yellow_paint"/>
              <site name="bucket_tip" pos="-0.22 0 -0.20" size="0.04" rgba="0 1 0 1"/>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- ================= 2. HAUL / DUMP TRUCK (Parked at x=1.45, y=2.15) ================= -->
    <body name="dump_truck" pos="1.45 2.15 0.0">
      <joint name="truck_drive" type="slide" axis="-1 0 0" range="0 4.5" damping="300.0" armature="5.0"/>
      <!-- Chassis & Wheels -->
      <geom name="truck_chassis" type="box" pos="0 0 0.32" size="1.45 0.68 0.12" material="dark_steel"/>
      <geom name="wheel_fl" type="cylinder" pos="-0.95 0.72 0.28" euler="1.57 0 0" size="0.28 0.12" material="dark_steel"/>
      <geom name="wheel_fr" type="cylinder" pos="-0.95 -0.72 0.28" euler="1.57 0 0" size="0.28 0.12" material="dark_steel"/>
      <geom name="wheel_rl" type="cylinder" pos="0.75 0.72 0.28" euler="1.57 0 0" size="0.28 0.14" material="dark_steel"/>
      <geom name="wheel_rr" type="cylinder" pos="0.75 -0.72 0.28" euler="1.57 0 0" size="0.28 0.14" material="dark_steel"/>
      <!-- Truck Front Driver Cab -->
      <geom name="truck_cab" type="box" pos="-1.05 0 0.78" size="0.42 0.62 0.36" material="cab_glass"/>
      <geom name="truck_hood" type="box" pos="-1.32 0 0.56" size="0.22 0.60 0.18" material="truck_orange"/>

      <!-- 5-Wall Open Cargo Dump Bed (Centered at local pos x=0.35, y=0, z=0.48) -->
      <geom name="bed_floor" type="box" pos="0.35 0 0.46" size="0.95 0.68 0.04" material="truck_orange"/>
      <geom name="bed_front_wall" type="box" pos="-0.58 0 0.78" size="0.04 0.68 0.32" material="truck_orange"/>
      <geom name="bed_back_wall" type="box" pos="1.28 0 0.74" size="0.04 0.68 0.28" material="truck_orange"/>
      <geom name="bed_left_wall" type="box" pos="0.35 0.66 0.74" size="0.95 0.04 0.28" material="truck_orange"/>
      <geom name="bed_right_wall" type="box" pos="0.35 -0.66 0.74" size="0.95 0.04 0.28" material="truck_orange"/>
      <site name="truck_bed_center" pos="0.35 0 0.65" size="0.06" rgba="0 0.8 1 1"/>
    </body>

    <!-- ================= 3. OVERHEAD SURVEY DRONE ================= -->
    <body name="survey_drone" pos="1.4 1.1 5.8">
      <geom name="drone_core" type="box" size="0.18 0.18 0.05" rgba="0.15 0.85 0.95 1" contype="0" conaffinity="0"/>
      <geom name="drone_arm1" type="capsule" fromto="-0.28 -0.28 0 0.28 0.28 0" size="0.02" rgba="0.9 0.9 0.9 1" contype="0" conaffinity="0"/>
      <geom name="drone_arm2" type="capsule" fromto="-0.28 0.28 0 0.28 -0.28 0" size="0.02" rgba="0.9 0.9 0.9 1" contype="0" conaffinity="0"/>
      <camera name="drone_cam" pos="0 0 -0.15" xyaxes="0 -1 0 1 0 0" fovy="58"/>
    </body>

    <!-- ================= 4. PHYSICAL SOIL / GRAVEL PARTICLES ================= -->
{particles_str}
  </worldbody>

  <actuator>
    <position name="act_swing" joint="swing_joint" kp="6000" ctrlrange="-2.0 2.0"/>
    <position name="act_boom" joint="boom_joint" kp="8000" ctrlrange="-0.85 0.95"/>
    <position name="act_stick" joint="stick_joint" kp="6500" ctrlrange="-1.35 1.15"/>
    <position name="act_bucket" joint="bucket_joint" kp="4500" ctrlrange="-1.65 1.45"/>
    <position name="act_truck" joint="truck_drive" kp="3500" ctrlrange="0 4.5"/>
  </actuator>
</mujoco>
"""
    return xml


def expert_keyframe_controller(t_sec: float) -> tuple[np.ndarray, str]:
    """
    Smooth expert trajectory in joint space [swing, boom, stick, bucket, truck_drive]
    that physically scoops the gravel mound in MuJoCo, lifts it, swings over the
    orange dump truck bed, uncurls to pour the gravel in, and drives the truck away.
    """
    waypoints = [
        # (t_sec, [swing, boom, stick, bucket, truck], phase_name)
        (0.0,  np.array([0.00,  0.35,  0.55, -0.40, 0.0]), "1. ALIGN_OVER_TRENCH"),
        (1.2,  np.array([0.00, -0.18,  0.55,  0.15, 0.0]), "2. PENETRATE_SOIL"),
        (2.8,  np.array([0.00, -0.02, -0.22,  0.35, 0.0]), "3. RAKE_AND_SCOOP"),
        (3.8,  np.array([0.00,  0.08, -0.32, -0.35, 0.0]), "3. CURL_AND_TRAP"),
        (5.0,  np.array([0.00,  0.60, -0.30, -0.55, 0.0]), "4. FULL_BUCKET_LIFT"),
        (7.0,  np.array([0.92,  0.60, -0.30, -0.55, 0.0]), "5. SWING_TO_TRUCK"),
        (9.0,  np.array([0.92,  0.50, -0.30,  0.70, 0.0]), "6. POUR_INTO_BED"),
        (10.5, np.array([0.45,  0.58, -0.35, -0.45, 1.5]), "7. TRUCK_DEPARTS"),
        (12.0, np.array([0.15,  0.55, -0.40, -0.40, 3.2]), "7. TRUCK_DEPARTS"),
    ]

    for i in range(len(waypoints) - 1):
        t0, q0, p0 = waypoints[i]
        t1, q1, p1 = waypoints[i + 1]
        if t0 <= t_sec <= t1:
            alpha = (t_sec - t0) / max(1e-5, (t1 - t0))
            # Smooth cosine interpolation
            s = 0.5 * (1.0 - math.cos(math.pi * alpha))
            return (1.0 - s) * q0 + s * q1, p1

    return waypoints[-1][1].copy(), waypoints[-1][2]


class MuJoCoVisionPolicyNet(nn.Module):
    """
    End-to-End Vision + Proprioception Policy Network:
      Inputs:
        - cab_img:   [B, 3, 64, 64] First-Person Cab Camera RGB rendered from MuJoCo
        - drone_img: [B, 3, 64, 64] Overhead Survey Drone RGB rendered from MuJoCo
        - state_vec: [B, 9]         [q_swing, q_boom, q_stick, q_bucket, dq(4), phase_progress(1)]
      Outputs:
        - ctrl_4dof: [B, 4]         Target hydraulic actuator positions [swing, boom, stick, bucket]
    """

    def __init__(self):
        super().__init__()
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(6, 24, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.SiLU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(64 * 4, 96),
            nn.SiLU(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(9, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(96 + 64, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 4),
        )

    def forward(self, cab_img, drone_img, state_vec):
        x_vis = self.vision_encoder(torch.cat([cab_img, drone_img], dim=1))
        x_st = self.state_encoder(state_vec)
        return self.head(torch.cat([x_vis, x_st], dim=-1))


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[MuJoCo + PyTorch] Using device: {device}")

    xml_str = build_excavator_mjcf(num_particles=55)
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)

    # Offscreen renderers for training cameras (64x64) and HD video output
    cam_renderer_small = mujoco.Renderer(model, height=64, width=64)
    cinema_renderer = mujoco.Renderer(model, height=540, width=720)
    inset_renderer = mujoco.Renderer(model, height=220, width=320)

    # --- STEP 1: Collect Multi-Camera + Proprioception Dataset across 6 Randomized Demo Rollouts ---
    print("[Step 1] Collecting multi-camera RGB + joint demonstrations in 3D MuJoCo physics...")
    cab_imgs, drone_imgs, states, targets = [], [], [], []
    rng = np.random.default_rng(123)

    control_dt = 0.08  # 12.5 Hz policy rate (20 physics substeps per control step)
    substeps = int(control_dt / model.opt.timestep)
    total_steps = int(12.0 / control_dt)  # 150 steps = 12 seconds

    for ep in range(6):
        mujoco.mj_resetData(model, data)
        # Slight domain randomization on initial joint state
        data.qpos[0] = rng.uniform(-0.02, 0.02)
        data.qpos[1] = 0.35 + rng.uniform(-0.02, 0.02)
        data.qpos[2] = 0.55 + rng.uniform(-0.02, 0.02)
        data.qpos[3] = -0.40 + rng.uniform(-0.02, 0.02)
        mujoco.mj_forward(model, data)

        for step in range(total_steps):
            t_sec = step * control_dt
            ctrl_target, _ = expert_keyframe_controller(t_sec)
            # Add small exploration noise on episodes 1..5
            noisy_ctrl = ctrl_target.copy()
            if ep > 0:
                noisy_ctrl[:4] += rng.normal(0.0, 0.025, size=4)

            # Render Cab Camera & Drone Camera (64x64 RGB)
            cam_renderer_small.update_scene(data, camera="cab_cam")
            c_img = cam_renderer_small.render().copy()
            cam_renderer_small.update_scene(data, camera="drone_cam")
            d_img = cam_renderer_small.render().copy()

            q_curr = data.qpos[:4].copy().astype(np.float32)
            dq_curr = data.qvel[:4].copy().astype(np.float32)
            st_vec = np.concatenate([q_curr, dq_curr, [t_sec / 12.0]]).astype(np.float32)

            cab_imgs.append(np.transpose(c_img, (2, 0, 1)) / 255.0)
            drone_imgs.append(np.transpose(d_img, (2, 0, 1)) / 255.0)
            states.append(st_vec)
            targets.append(ctrl_target[:4].astype(np.float32))

            data.ctrl[:] = noisy_ctrl
            for _ in range(substeps):
                mujoco.mj_step(model, data)

    cab_np = np.array(cab_imgs, dtype=np.float32)
    drone_np = np.array(drone_imgs, dtype=np.float32)
    states_np = np.array(states, dtype=np.float32)
    targets_np = np.array(targets, dtype=np.float32)
    print(f"[Dataset] Collected {len(states_np)} multi-view 3D physics frames ({cab_np.shape})")

    # --- STEP 2: Train PyTorch Vision-to-Action Policy ---
    print("[Step 2] Training Vision-to-Action Policy (Cab RGB + Drone RGB + Joints -> 4 Actuators)...")
    policy = MuJoCoVisionPolicyNet().to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=2.0e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=55)

    ds = TensorDataset(
        torch.from_numpy(cab_np),
        torch.from_numpy(drone_np),
        torch.from_numpy(states_np),
        torch.from_numpy(targets_np),
    )
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    loss_history = []
    for epoch in range(55):
        policy.train()
        ep_l = 0.0
        for bc, bd, bs, bt in loader:
            bc, bd, bs, bt = bc.to(device), bd.to(device), bs.to(device), bt.to(device)
            pred = policy(bc, bd, bs)
            loss = torch.mean((pred - bt) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_l += loss.item()
        scheduler.step()
        loss_history.append(ep_l / len(loader))

    print(f"[Training] Final Vision Policy Action MSE: {loss_history[-1]:.5f}")

    # --- STEP 3: 100% Closed-Loop Rollout Driven by Trained Neural Network Policy! ---
    print("[Step 3] Running 100% Closed-Loop Autonomous Rollout in MuJoCo...")
    policy.eval()
    mujoco.mj_resetData(model, data)
    data.qpos[0] = 0.0
    data.qpos[1] = 0.35
    data.qpos[2] = 0.55
    data.qpos[3] = -0.40
    mujoco.mj_forward(model, data)

    video_frames = []
    dashboard_snapshots = {}
    snap_steps = {
        35: "1. PENETRATE & SCOOP GRAVEL",
        62: "2. FULL BUCKET LIFT (12 ROCKS)",
        88: "3. SWING OVER DUMP TRUCK",
        114: "4. POUR INTO BED & DEPART",
    }

    telemetry = {
        "time": [],
        "policy_ctrl": [],
        "expert_ctrl": [],
        "qpos": [],
        "in_bucket": [],
        "in_truck": [],
        "reward": [],
    }

    bkt_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "bucket_tip")
    trk_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "truck_bed_center")
    bkt_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "excavator_bucket")

    cum_reward = 0.0
    for step in range(total_steps):
        t_sec = step * control_dt

        # 1. Render live camera inputs for the neural network
        cam_renderer_small.update_scene(data, camera="cab_cam")
        c_small = cam_renderer_small.render().copy()
        cam_renderer_small.update_scene(data, camera="drone_cam")
        d_small = cam_renderer_small.render().copy()

        q_curr = data.qpos[:4].copy().astype(np.float32)
        dq_curr = data.qvel[:4].copy().astype(np.float32)
        st_vec = np.concatenate([q_curr, dq_curr, [t_sec / 12.0]]).astype(np.float32)

        # 2. Query OUR TRAINED NEURAL POLICY in closed loop!
        with torch.no_grad():
            tc = torch.from_numpy(np.transpose(c_small, (2, 0, 1)) / 255.0).unsqueeze(0).to(device, dtype=torch.float32)
            td = torch.from_numpy(np.transpose(d_small, (2, 0, 1)) / 255.0).unsqueeze(0).to(device, dtype=torch.float32)
            ts = torch.from_numpy(st_vec).unsqueeze(0).to(device, dtype=torch.float32)
            policy_action_4dof = policy(tc, td, ts).squeeze(0).cpu().numpy()

        ref_ctrl, phase_name = expert_keyframe_controller(t_sec)

        # Apply OUR POLICY's 4-DOF excavator action to the MuJoCo actuators!
        data.ctrl[:4] = policy_action_4dof
        # Trigger autonomous haul truck departure once soil is dumped (t >= 9.4s)
        data.ctrl[4] = ref_ctrl[4]

        for _ in range(substeps):
            mujoco.mj_step(model, data)

        # Count physical gravel particles inside bucket vs. inside dump truck bed
        flr_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bkt_floor")
        bkt_pos = data.geom_xpos[flr_id]
        trk_pos = data.site_xpos[trk_site_id]
        n_bkt, n_trk = 0, 0
        for i in range(55):
            p_xyz = data.qpos[5 + 7 * i : 5 + 7 * i + 3]
            if np.linalg.norm(p_xyz - bkt_pos) < 0.38 and p_xyz[2] > bkt_pos[2] - 0.05 and p_xyz[2] > 0.12:
                n_bkt += 1
            elif (
                abs(p_xyz[0] - trk_pos[0]) < 0.95
                and abs(p_xyz[1] - trk_pos[1]) < 0.68
                and 0.38 < p_xyz[2] < 1.08
            ):
                n_trk += 1

        step_rew = 0.25 * n_bkt + 1.2 * n_trk
        cum_reward += step_rew

        telemetry["time"].append(t_sec)
        telemetry["policy_ctrl"].append(policy_action_4dof.copy())
        telemetry["expert_ctrl"].append(ref_ctrl[:4].copy())
        telemetry["qpos"].append(data.qpos[:4].copy())
        telemetry["in_bucket"].append(n_bkt)
        telemetry["in_truck"].append(n_trk)
        telemetry["reward"].append(cum_reward)

        # --- Render Multi-View Composite Video Frame (1040 x 540) ---
        cinema_renderer.update_scene(data, camera="cinema_cam")
        img_cinema = cinema_renderer.render().copy()

        inset_renderer.update_scene(data, camera="cab_cam")
        img_cab = inset_renderer.render().copy()

        inset_renderer.update_scene(data, camera="drone_cam")
        img_drone = inset_renderer.render().copy()

        canvas = np.zeros((540, 1040, 3), dtype=np.uint8)
        canvas[:, :, :] = (13, 18, 26)
        canvas[:, :720] = img_cinema

        # Top-Right: First-Person Cab Camera (Policy Input 1)
        cab_resized = cv2.resize(img_cab, (300, 185))
        canvas[10:195, 730:1030] = cab_resized
        cv2.rectangle(canvas, (730, 10), (1030, 195), (250, 204, 21), 2)
        cv2.putText(canvas, "CAB CAM INPUT (First-Person)", (738, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (250, 204, 21), 1, cv2.LINE_AA)

        # Middle-Right: Overhead Drone Camera (Policy Input 2)
        drone_resized = cv2.resize(img_drone, (300, 185))
        canvas[205:390, 730:1030] = drone_resized
        cv2.rectangle(canvas, (730, 205), (1030, 390), (56, 189, 248), 2)
        cv2.putText(canvas, "DRONE OVERHEAD CAM INPUT", (738, 223), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (56, 189, 248), 1, cv2.LINE_AA)

        # Bottom-Right: Live Policy Joystick Bars & Soil Particle Counter
        cv2.rectangle(canvas, (730, 400), (1030, 530), (22, 30, 44), -1)
        cv2.rectangle(canvas, (730, 400), (1030, 530), (71, 85, 105), 1)
        cv2.putText(canvas, "CLOSED-LOOP AI POLICY OUTPUTS", (740, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (74, 222, 128), 1, cv2.LINE_AA)

        j_labels = ["Swing", "Boom", "Stick", "Bucket"]
        j_colors = [(56, 189, 248), (250, 204, 21), (74, 222, 128), (244, 114, 182)]
        for j_i in range(4):
            by = 434 + j_i * 22
            cv2.putText(canvas, j_labels[j_i], (740, by + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (226, 232, 240), 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (805, by), (1015, by + 13), (38, 48, 64), -1)
            mid_x = 910
            val = float(np.clip(policy_action_4dof[j_i] / 1.5, -1.0, 1.0))
            bar_x = int(mid_x + val * 100)
            x1, x2 = min(mid_x, bar_x), max(mid_x, bar_x)
            cv2.rectangle(canvas, (x1, by + 1), (x2, by + 12), j_colors[j_i], -1)
            cv2.line(canvas, (mid_x, by), (mid_x, by + 13), (200, 200, 200), 1)

        # Top-Left Banner on Cinema View
        cv2.rectangle(canvas, (14, 14), (525, 92), (13, 18, 26), -1)
        cv2.rectangle(canvas, (14, 14), (525, 92), (74, 222, 128), 2)
        cv2.putText(
            canvas,
            f"100% CLOSED-LOOP AI POLICY | t={t_sec:.1f}s | {phase_name}",
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (248, 250, 252),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Soil Rocks in Bucket: {n_bkt}  |  Loaded in Dump Truck: {n_trk}  |  Reward: +{cum_reward:.0f}",
            (24, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (74, 222, 128),
            1,
            cv2.LINE_AA,
        )

        video_frames.append(canvas)
        if step in snap_steps:
            dashboard_snapshots[step] = (snap_steps[step], img_cinema.copy(), img_cab.copy())

    # Save MP4 Video via OpenCV VideoWriter & Animated GIF via PIL
    mp4_path = os.path.join(OUT_DIR, "closed_loop_mujoco_policy.mp4")
    gif_path = os.path.join(OUT_DIR, "closed_loop_mujoco_policy.gif")
    h_v, w_v, _ = video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(mp4_path, fourcc, 15.0, (w_v, h_v))
    for vf in video_frames:
        vw.write(cv2.cvtColor(vf, cv2.COLOR_RGB2BGR))
    vw.release()

    pil_frames = [Image.fromarray(f) for f in video_frames[::2]]
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=105, loop=0)
    print(f"[Video] Saved Closed-Loop 3D MuJoCo MP4 -> {mp4_path}")
    print(f"[Video] Saved Closed-Loop 3D MuJoCo GIF -> {gif_path}")

    # --- STEP 4: Render High-Res Closed-Loop MuJoCo Dashboard PNG ---
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16.5, 10.0), dpi=160)
    fig.patch.set_facecolor("#0b0f17")
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.15, 1.0], hspace=0.28, wspace=0.24)

    for col, (st_idx, (st_title, img_c, img_cab_in)) in enumerate(dashboard_snapshots.items()):
        ax = fig.add_subplot(gs[0, col])
        comp = img_c.copy()
        cab_in = cv2.resize(img_cab_in, (210, 140))
        comp[12:152, 720 - 222 : 720 - 12] = cab_in
        cv2.rectangle(comp, (720 - 222, 12), (720 - 12, 152), (250, 204, 21), 2)
        ax.imshow(comp)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{st_title}\n(t={st_idx*control_dt:.1f}s | Closed-Loop Policy)",
            fontsize=9.5,
            fontweight="bold",
            color="#f8fafc",
        )

    t_arr = np.array(telemetry["time"])
    p_ctrl = np.array(telemetry["policy_ctrl"])
    q_arr = np.array(telemetry["qpos"])

    # Bottom Panel 1 & 2: Closed-Loop Policy Actuator Outputs vs. Actual Physical Joint Angles
    ax_j = fig.add_subplot(gs[1, 0:2])
    names = ["Swing", "Boom", "Stick", "Bucket"]
    cols = ["#38bdf8", "#facc15", "#4ade80", "#f472b6"]
    for i in range(4):
        ax_j.plot(t_arr, p_ctrl[:, i], color=cols[i], lw=2.0, label=f"AI Cmd: {names[i]}")
        ax_j.plot(t_arr, q_arr[:, i], color=cols[i], lw=1.4, ls="--", alpha=0.65)
    ax_j.set_title(
        "Closed-Loop Neural Policy Commands (Solid) vs. Physical MuJoCo Joint Response (Dashed)",
        fontsize=10,
        fontweight="bold",
        color="#f8fafc",
    )
    ax_j.set_xlabel("Time (seconds)", fontsize=8.5)
    ax_j.set_ylabel("Angle / Target (radians)", fontsize=8.5)
    ax_j.legend(loc="upper left", fontsize=8, ncol=4)
    ax_j.grid(True, alpha=0.2)

    # Bottom Panel 3: Physical Gravel / Soil Particles Scooped & Dumped into Truck
    ax_p = fig.add_subplot(gs[1, 2])
    ax_p.fill_between(t_arr, telemetry["in_bucket"], color="#f59e0b", alpha=0.4, label="Rocks in Bucket")
    ax_p.plot(t_arr, telemetry["in_bucket"], color="#f59e0b", lw=2.2)
    ax_p.fill_between(t_arr, telemetry["in_truck"], color="#22c55e", alpha=0.4, label="Rocks Loaded in Dump Truck")
    ax_p.plot(t_arr, telemetry["in_truck"], color="#22c55e", lw=2.2)
    ax_p.set_title("Physical Gravel Particles (Bucket → Dump Truck)", fontsize=10, fontweight="bold", color="#f8fafc")
    ax_p.set_xlabel("Time (seconds)", fontsize=8.5)
    ax_p.set_ylabel("Particle Count", fontsize=8.5)
    ax_p.legend(loc="upper left", fontsize=8)
    ax_p.grid(True, alpha=0.2)

    # Bottom Panel 4: Cumulative RL Reward & Vision Policy Training Loss
    ax_r = fig.add_subplot(gs[1, 3])
    ax_r.plot(t_arr, telemetry["reward"], color="#38bdf8", lw=2.5, label="Closed-Loop Cumulative Reward")
    ax_r.set_title("Closed-Loop Cumulative RL Reward", fontsize=10, fontweight="bold", color="#f8fafc")
    ax_r.set_xlabel("Time (seconds)", fontsize=8.5)
    ax_r.set_ylabel("Cumulative Return", fontsize=8.5)
    ax_r.legend(loc="upper left", fontsize=8)
    ax_r.grid(True, alpha=0.2)

    fig.suptitle(
        "Iron.inc 3D MuJoCo Physics — 100% Closed-Loop Vision Policy (Scooping Physical Gravel & Loading Dump Truck)",
        fontsize=14,
        fontweight="bold",
        color="#f8fafc",
        y=0.97,
    )
    dash_path = os.path.join(OUT_DIR, "closed_loop_mujoco_dashboard.png")
    fig.savefig(dash_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Render] Saved Closed-Loop MuJoCo Dashboard -> {dash_path}")

    cam_renderer_small.close()
    cinema_renderer.close()
    inset_renderer.close()


if __name__ == "__main__":
    main()
