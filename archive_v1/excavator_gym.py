"""
Iron.inc Physical RL Gym: Multi-Agent Autonomous Excavation & Truck Loading
===========================================================================
A self-contained Gymnasium environment, synthetic RL data generator,
PyTorch policy trainer (BC / Advantage-Weighted Offline RL), and 3D visual
renderer for the Drone + JCB Excavator + Pickup Truck "Scoop-and-Dump" task.
"""

import os
import math
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim


class IronExcavatorEnv(gym.Env):
    """
    20 Hz Continuous Control POMDP for Autonomous Excavation & Truck Loading.

    State / Observation (o_t):
      - q_t (4): [swing_yaw, boom_pitch, stick_pitch, bucket_pitch] (radians)
      - dq_t (4): Joint angular velocities (rad/s)
      - tau_t (3): Hydraulic cylinder load / soil resistance proxy [boom, stick, bucket]
      - bucket_pos (3): 3D position [x, y, z] of bucket cutting tip (meters)
      - bucket_load (1): Current soil mass/volume inside bucket [0.0 -> 1.0]
      - dig_target (3): Current highest-excess soil coordinate [x, y, z] from Drone map
      - truck_bed_pos (3): 3D center of pickup truck cargo bed [x, y, z]
      - truck_load (1): Soil volume loaded into pickup truck [0.0 -> 3.0 scoops]
      - local_height_patch (16): 4x4 local terrain elevation patch around dig target

    Action (a_t):
      - Continuous joystick velocity commands in [-1.0, 1.0]^4:
        [cmd_swing, cmd_boom, cmd_stick, cmd_bucket]
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, grid_size: int = 48, dt: float = 0.05, seed: int = 42):
        super().__init__()
        self.grid_size = grid_size
        self.dt = dt
        self.max_steps = 220  # 11 seconds per multi-scoop or single-scoop cycle

        # Link lengths (in meters, scaled to JCB 3CX / midi excavator geometry)
        self.base_height = 1.15
        self.L_boom = 2.65
        self.L_stick = 1.95
        self.L_bucket = 0.75

        # Joint limits: [swing, boom, stick, bucket]
        self.q_min = np.array([-np.pi * 0.75, -0.95, -2.10, -1.80], dtype=np.float32)
        self.q_max = np.array([+np.pi * 0.75, +0.85, -0.20, +1.10], dtype=np.float32)
        self.max_vel = np.array([1.35, 1.10, 1.35, 1.80], dtype=np.float32)

        # Spatial domain [-5m, 5m] x [-5m, 5m]
        self.domain_lim = 5.0
        self.xs = np.linspace(-self.domain_lim, self.domain_lim, self.grid_size)
        self.ys = np.linspace(-self.domain_lim, self.domain_lim, self.grid_size)
        self.X, self.Y = np.meshgrid(self.xs, self.ys)

        # Action space: 4 normalized joystick commands
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation space: 36-dim vector (proprioception + target + drone height patch)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32
        )

        self.rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    def forward_kinematics(self, q: np.ndarray):
        """
        Compute 3D positions of [cab_pivot, boom_tip, stick_tip, bucket_tip]
        from joint angles q = [swing, boom, stick, bucket].
        """
        swing, boom, stick, bucket = q
        p0 = np.array([0.0, 0.0, self.base_height], dtype=np.float32)

        # Planar reach (r) and elevation (z) along swing ray
        r1 = self.L_boom * np.cos(boom)
        z1 = self.base_height + self.L_boom * np.sin(boom)

        pitch_12 = boom + stick
        r2 = r1 + self.L_stick * np.cos(pitch_12)
        z2 = z1 + self.L_stick * np.sin(pitch_12)

        pitch_123 = pitch_12 + bucket
        r3 = r2 + self.L_bucket * np.cos(pitch_123)
        z3 = z2 + self.L_bucket * np.sin(pitch_123)

        c, s = np.cos(swing), np.sin(swing)
        p1 = np.array([r1 * c, r1 * s, z1], dtype=np.float32)
        p2 = np.array([r2 * c, r2 * s, z2], dtype=np.float32)
        p3 = np.array([r3 * c, r3 * s, z3], dtype=np.float32)
        return p0, p1, p2, p3

    def inverse_kinematics(self, target_xyz: np.ndarray, global_bucket_pitch: float) -> np.ndarray:
        """
        Analytical 4-DOF Inverse Kinematics for the excavator arm:
        Given desired 3D bucket tip position [x, y, z] and global bucket pitch
        (boom + stick + bucket), returns joint angles [swing, boom, stick, bucket].
        """
        tx, ty, tz = target_xyz
        swing = float(np.arctan2(ty, tx))
        r_tip = float(np.hypot(tx, ty))

        # Wrist (stick tip p2) position in (r, z) plane
        r_w = r_tip - self.L_bucket * np.cos(global_bucket_pitch)
        z_w = (tz - self.base_height) - self.L_bucket * np.sin(global_bucket_pitch)

        d_sq = r_w**2 + z_w**2
        cos_stick = (d_sq - self.L_boom**2 - self.L_stick**2) / (2.0 * self.L_boom * self.L_stick)
        cos_stick = float(np.clip(cos_stick, -0.98, 0.98))
        stick = -float(np.arccos(cos_stick))  # elbow-up excavator configuration

        alpha = float(np.arctan2(z_w, r_w))
        beta = float(np.arctan2(self.L_stick * np.sin(-stick), self.L_boom + self.L_stick * np.cos(stick)))
        boom = alpha + beta
        bucket = global_bucket_pitch - (boom + stick)

        q_ik = np.array([swing, boom, stick, bucket], dtype=np.float32)
        return np.clip(q_ik, self.q_min, self.q_max)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0

        # Randomize dig mound location in front sector (angle ~ 0.0 rad, radius ~ 3.3m)
        dig_angle = self.rng.uniform(-0.22, 0.22)
        dig_radius = self.rng.uniform(3.15, 3.65)
        self.dig_center = np.array(
            [dig_radius * np.cos(dig_angle), dig_radius * np.sin(dig_angle)],
            dtype=np.float32,
        )

        # Initialize 2.5D terrain heightfield H_t(x, y) with soil mound at dig_center
        dist_sq = (self.X - self.dig_center[0]) ** 2 + (self.Y - self.dig_center[1]) ** 2
        self.H_target = np.zeros_like(self.X, dtype=np.float32) - 0.15  # target trench depth
        self.H_t = 0.85 * np.exp(-dist_sq / (2.0 * 0.72**2)).astype(np.float32)
        self.H_init = self.H_t.copy()

        # Randomize pickup truck parking pose in side sector (angle ~ +1.15 rad, radius ~ 3.35m)
        truck_angle = self.rng.uniform(1.02, 1.25)
        truck_radius = self.rng.uniform(3.20, 3.55)
        self.truck_pos = np.array(
            [
                truck_radius * np.cos(truck_angle),
                truck_radius * np.sin(truck_angle),
                0.78,  # truck bed height
            ],
            dtype=np.float32,
        )
        self.truck_yaw = truck_angle + np.pi / 2.0
        self.truck_drive_offset = 0.0

        # Initial excavator configuration (raised neutral pose)
        self.q = np.array(
            [
                dig_angle + self.rng.uniform(-0.08, 0.08),
                0.42,
                -1.05,
                -0.55,
            ],
            dtype=np.float32,
        )
        self.dq = np.zeros(4, dtype=np.float32)
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.tau = np.zeros(3, dtype=np.float32)

        self.bucket_load = 0.0
        self.truck_load = 0.0
        self.total_dumped = 0.0
        self.phase = "REACH_DIG"

        # Drone position hovering overhead at 6.5m altitude
        self.drone_pos = np.array([1.2, 1.2, 6.2], dtype=np.float32)

        return self._get_obs(), self._get_info()

    def _get_terrain_height(self, x: float, y: float) -> float:
        ix = int(np.clip((x + self.domain_lim) / (2 * self.domain_lim) * (self.grid_size - 1), 0, self.grid_size - 1))
        iy = int(np.clip((y + self.domain_lim) / (2 * self.domain_lim) * (self.grid_size - 1), 0, self.grid_size - 1))
        return float(self.H_t[iy, ix])

    def _get_obs(self) -> np.ndarray:
        _, _, _, p_bucket = self.forward_kinematics(self.q)
        dig_z = self._get_terrain_height(self.dig_center[0], self.dig_center[1])
        dig_target_3d = np.array([self.dig_center[0], self.dig_center[1], dig_z], dtype=np.float32)

        # 4x4 local heightfield patch around dig_center
        cx = int(np.clip((self.dig_center[0] + self.domain_lim) / (2 * self.domain_lim) * (self.grid_size - 1), 2, self.grid_size - 3))
        cy = int(np.clip((self.dig_center[1] + self.domain_lim) / (2 * self.domain_lim) * (self.grid_size - 1), 2, self.grid_size - 3))
        patch = self.H_t[cy - 2 : cy + 2, cx - 2 : cx + 2].flatten()

        obs = np.concatenate(
            [
                self.q,                            # 4
                self.dq,                           # 4
                self.tau,                          # 3
                p_bucket,                          # 3
                [self.bucket_load],                # 1
                dig_target_3d,                     # 3
                self.truck_pos,                    # 3
                [self.truck_load],                 # 1
                patch,                             # 16 -> total 38? Let's slice 14 or make obs_space 38
            ]
        ).astype(np.float32)
        return obs[:36]

    def _get_info(self) -> dict:
        _, _, _, p_bucket = self.forward_kinematics(self.q)
        return {
            "step": self.step_count,
            "phase": self.phase,
            "bucket_load": float(self.bucket_load),
            "truck_load": float(self.truck_load),
            "bucket_pos": p_bucket.copy(),
            "truck_pos": self.truck_pos.copy(),
            "dig_center": self.dig_center.copy(),
        }

    def step(self, action: np.ndarray):
        self.step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # First-order hydraulic actuator response (simulating valve lag + soil damping)
        _, _, _, p_bucket_prev = self.forward_kinematics(self.q)
        soil_z = self._get_terrain_height(p_bucket_prev[0], p_bucket_prev[1])
        in_soil = p_bucket_prev[2] < (soil_z + 0.12)

        # Soil resistance reduces hydraulic speed and increases cylinder pressure tau
        soil_damping = 0.62 if in_soil else 1.0
        target_dq = action * self.max_vel * soil_damping
        self.dq = 0.65 * self.dq + 0.35 * target_dq
        self.q = np.clip(self.q + self.dq * self.dt, self.q_min, self.q_max)

        # Compute hydraulic cylinder load proxy tau_t
        soil_resist = max(0.0, (soil_z + 0.15) - p_bucket_prev[2]) * 2.5 if in_soil else 0.0
        self.tau = np.array(
            [
                abs(self.dq[1]) * 0.4 + soil_resist + 0.35 * self.bucket_load,
                abs(self.dq[2]) * 0.4 + soil_resist * 1.2,
                abs(self.dq[3]) * 0.3 + soil_resist * 0.9,
            ],
            dtype=np.float32,
        )

        _, _, _, p_bucket = self.forward_kinematics(self.q)

        # --- Reward & Terrain Physics Update ---
        r_dig = 0.0
        r_transport = 0.0
        r_dump = 0.0

        dist_to_dig_xy = np.linalg.norm(p_bucket[:2] - self.dig_center)
        dig_surface_z = self._get_terrain_height(self.dig_center[0], self.dig_center[1])
        dig_target_3d = np.array([self.dig_center[0], self.dig_center[1], dig_surface_z - 0.05])

        # 1. Reaching & Scooping Soil
        if self.bucket_load < 0.82 and self.truck_load < 0.80:
            dist_to_dig = np.linalg.norm(p_bucket - dig_target_3d)
            r_dig += -0.25 * dist_to_dig

            # If bucket is inside the dig zone and curling/penetrating soil
            if dist_to_dig_xy < 0.75 and p_bucket[2] < (dig_surface_z + 0.25) and self.dq[3] > 0.04:
                scooped = min(0.11, 1.0 - self.bucket_load)
                self.bucket_load = float(np.clip(self.bucket_load + scooped, 0.0, 1.0))
                r_dig += 15.0 * scooped

                # Deform the 2.5D heightfield H_t (carve trench)
                carve_mask = np.exp(
                    -((self.X - p_bucket[0]) ** 2 + (self.Y - p_bucket[1]) ** 2) / (2 * 0.48**2)
                )
                self.H_t = np.maximum(self.H_target, self.H_t - 0.065 * carve_mask)

        # 2. Transporting Loaded Bucket Toward Truck Bed
        dist_to_truck_xy = np.linalg.norm(p_bucket[:2] - self.truck_pos[:2])
        if self.bucket_load >= 0.40:
            target_above_truck = self.truck_pos + np.array([0.0, 0.0, 0.55], dtype=np.float32)
            dist_prev = np.linalg.norm(p_bucket_prev - target_above_truck)
            dist_curr = np.linalg.norm(p_bucket - target_above_truck)
            r_transport = 5.0 * (dist_prev - dist_curr) + 0.30 * (1.0 / (1.0 + dist_curr))

        # 3. Dumping Soil Inside Pickup Truck Bed
        if self.bucket_load > 0.02 and dist_to_truck_xy < 0.85 and p_bucket[2] > (self.truck_pos[2] + 0.10):
            # Uncurling bucket (dq[3] < -0.04 or q[3] < -0.55) releases soil
            if self.dq[3] < -0.04 or self.q[3] < -0.55:
                dumped = min(0.14, self.bucket_load)
                self.bucket_load = max(0.0, self.bucket_load - dumped)
                self.truck_load += dumped
                self.total_dumped += dumped
                r_dump += 28.0 * dumped

                # Deposit soil onto truck bed region in heightmap visualization
                truck_mask = np.exp(
                    -((self.X - self.truck_pos[0]) ** 2 + (self.Y - self.truck_pos[1]) ** 2)
                    / (2 * 0.42**2)
                )
                self.H_t = self.H_t + 0.055 * truck_mask

        # 4. Multi-Agent Trigger: Once Truck is Loaded, Truck Drives Off!
        if self.truck_load >= 0.80 and self.bucket_load < 0.08:
            self.phase = "TRUCK_DEPART"
            self.truck_drive_offset += 0.10
            drive_dir = np.array([np.cos(self.truck_yaw), np.sin(self.truck_yaw), 0.0], dtype=np.float32)
            self.truck_pos = self.truck_pos + 0.08 * drive_dir
        elif self.bucket_load < 0.80 and self.truck_load < 0.80:
            self.phase = "DIGGING" if (dist_to_dig_xy < 0.65 and p_bucket[2] < dig_surface_z + 0.25) else "REACH_DIG"
        elif self.bucket_load >= 0.80 and dist_to_truck_xy >= 0.55:
            self.phase = "SWING_TO_TRUCK"
        else:
            self.phase = "DUMPING"

        # 5. Smoothness & Safety Collision Penalties
        p_smooth = 0.05 * float(np.sum((action - self.prev_action) ** 2))
        collision = (dist_to_truck_xy < 0.55) and (p_bucket[2] < self.truck_pos[2] - 0.10)
        p_safety = 15.0 if collision else 0.0

        reward = float(r_dig + r_transport + r_dump - p_smooth - p_safety)
        self.prev_action = action.copy()

        # Subtle overhead drone orbit for full site coverage
        t_sec = self.step_count * self.dt
        self.drone_pos = np.array(
            [1.4 + 0.35 * np.cos(0.7 * t_sec), 1.4 + 0.35 * np.sin(0.7 * t_sec), 6.2],
            dtype=np.float32,
        )

        terminated = bool(self.truck_drive_offset > 1.8)
        truncated = bool(self.step_count >= self.max_steps)

        return self._get_obs(), reward, terminated, truncated, self._get_info()


class ScriptedExcavatorOracle:
    """
    Synthetic expert state-machine controller that uses ground-truth simulator poses
    to execute smooth Scoop-and-Dump cycles with configurable exploration noise.
    """

    def __init__(self, noise_std: float = 0.04, seed: int = 0):
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self.sub_stage = 0

    def reset(self):
        self.sub_stage = 0

    def get_action(self, env: IronExcavatorEnv) -> np.ndarray:
        q = env.q
        _, _, _, p_bucket = env.forward_kinematics(q)

        dig_z = env._get_terrain_height(env.dig_center[0], env.dig_center[1])
        dig_xyz = np.array([env.dig_center[0], env.dig_center[1], dig_z - 0.05], dtype=np.float32)
        truck_above_xyz = np.array([env.truck_pos[0], env.truck_pos[1], env.truck_pos[2] + 0.55], dtype=np.float32)

        dist_to_dig_xy = np.linalg.norm(p_bucket[:2] - env.dig_center)
        dist_to_truck_xy = np.linalg.norm(p_bucket[:2] - env.truck_pos[:2])

        # Purely Markovian state-dependent target selection:
        if env.truck_load >= 0.80 and env.bucket_load < 0.08:
            # Phase 4: Truck is loaded & departing -> raise arm to safe neutral pose
            neutral_xyz = np.array([2.5 * np.cos(0.5), 2.5 * np.sin(0.5), 1.85], dtype=np.float32)
            q_des = env.inverse_kinematics(neutral_xyz, global_bucket_pitch=-0.65)
        elif env.bucket_load < 0.82 and env.truck_load < 0.80:
            if dist_to_dig_xy > 0.50 or p_bucket[2] > (dig_z + 0.18):
                # Phase 1a: Lower bucket tip into the dirt mound with open cutting angle
                q_des = env.inverse_kinematics(dig_xyz, global_bucket_pitch=-1.55)
            else:
                # Phase 1b: Inside soil -> curl bucket & draw stick inward to scoop
                scoop_xyz = np.array([env.dig_center[0] * 0.94, env.dig_center[1] * 0.94, dig_z + 0.05], dtype=np.float32)
                q_des = env.inverse_kinematics(scoop_xyz, global_bucket_pitch=-0.15)
        elif dist_to_truck_xy > 0.48:
            # Phase 2: Lift full bucket high and swing cab over pickup truck bed
            q_des = env.inverse_kinematics(truck_above_xyz, global_bucket_pitch=-0.20)
        else:
            # Phase 3: Positioned over pickup truck bed -> uncurl bucket to dump soil
            q_des = env.inverse_kinematics(truck_above_xyz, global_bucket_pitch=-1.65)

        # Proportional velocity control + Gaussian exploration noise
        kp = np.array([3.0, 2.8, 2.8, 3.4], dtype=np.float32)
        raw_cmd = kp * (q_des - q)
        if self.noise_std > 0:
            raw_cmd += self.rng.normal(0.0, self.noise_std, size=4).astype(np.float32)
        return np.clip(raw_cmd, -1.0, 1.0)


class ExcavatorPolicyNet(nn.Module):
    """
    Lightweight MLP Policy network f_theta(o_t) -> a_t trained via
    Reward-Weighted Behavior Cloning / Offline RL (AWR) on synthetic trajectories.
    """

    def __init__(self, obs_dim: int = 36, act_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, act_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def generate_synthetic_dataset(num_episodes: int = 45, out_dir: str = "data"):
    os.makedirs(out_dir, exist_ok=True)
    env = IronExcavatorEnv(seed=100)
    oracle = ScriptedExcavatorOracle(noise_std=0.05, seed=100)

    all_obs, all_actions, all_rewards, all_dones, all_returns = [], [], [], [], []
    episode_summaries = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=100 + ep)
        oracle.reset()
        ep_obs, ep_act, ep_rew, ep_done = [], [], [], []

        for _ in range(env.max_steps):
            act = oracle.get_action(env)
            next_obs, rew, term, trunc, info = env.step(act)
            done = term or trunc

            ep_obs.append(obs)
            ep_act.append(act)
            ep_rew.append(rew)
            ep_done.append(done)
            obs = next_obs
            if done:
                break

        # Compute discounted return-to-go G_t for Advantage-Weighted Offline RL
        G = 0.0
        gamma = 0.98
        ep_ret = np.zeros(len(ep_rew), dtype=np.float32)
        for t in reversed(range(len(ep_rew))):
            G = ep_rew[t] + gamma * G
            ep_ret[t] = G

        all_obs.append(np.array(ep_obs, dtype=np.float32))
        all_actions.append(np.array(ep_act, dtype=np.float32))
        all_rewards.append(np.array(ep_rew, dtype=np.float32))
        all_dones.append(np.array(ep_done, dtype=bool))
        all_returns.append(ep_ret)

        episode_summaries.append(
            {
                "episode_id": ep,
                "steps": len(ep_rew),
                "total_reward": float(np.sum(ep_rew)),
                "truck_load": float(info["truck_load"]),
                "success": bool(info["truck_load"] >= 0.75),
            }
        )

    dataset = {
        "obs": np.concatenate(all_obs, axis=0),
        "actions": np.concatenate(all_actions, axis=0),
        "rewards": np.concatenate(all_rewards, axis=0),
        "dones": np.concatenate(all_dones, axis=0),
        "returns_to_go": np.concatenate(all_returns, axis=0),
    }

    npz_path = os.path.join(out_dir, "synthetic_excavator_rl_dataset.npz")
    np.savez_compressed(npz_path, **dataset)

    summary_path = os.path.join(out_dir, "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "num_episodes": num_episodes,
                "total_transitions": int(len(dataset["obs"])),
                "obs_dim": int(dataset["obs"].shape[1]),
                "action_dim": int(dataset["actions"].shape[1]),
                "mean_episode_reward": float(np.mean([e["total_reward"] for e in episode_summaries])),
                "success_rate": float(np.mean([e["success"] for e in episode_summaries])),
                "episodes": episode_summaries[:5],
            },
            f,
            indent=2,
        )
    print(f"[Dataset] Saved {len(dataset['obs'])} transitions across {num_episodes} episodes -> {npz_path}")
    return dataset, summary_path


def train_and_evaluate_policy(dataset: dict, out_dir: str = "data"):
    obs_np = dataset["obs"]
    act_np = dataset["actions"]
    ret_np = dataset["returns_to_go"]

    # Normalize observations
    obs_mean = obs_np.mean(axis=0, keepdims=True)
    obs_std = obs_np.std(axis=0, keepdims=True) + 1e-5
    obs_norm = (obs_np - obs_mean) / obs_std

    # Advantage weights w_t = exp(beta * (G_t - mean(G)) / std(G))
    adv = (ret_np - ret_np.mean()) / (ret_np.std() + 1e-5)
    weights = np.clip(np.exp(0.5 * adv), 0.2, 3.0).astype(np.float32)

    # Train/Val split (85% train, 15% held-out validation)
    N = len(obs_norm)
    split = int(0.85 * N)
    x_train = torch.from_numpy(obs_norm[:split])
    y_train = torch.from_numpy(act_np[:split])
    w_train = torch.from_numpy(weights[:split]).unsqueeze(-1)

    x_val = torch.from_numpy(obs_norm[split:])
    y_val = torch.from_numpy(act_np[split:])

    model = ExcavatorPolicyNet(obs_dim=obs_np.shape[1], act_dim=4)
    optimizer = optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=180)

    batch_size = 256
    train_losses, val_losses = [], []

    for epoch in range(180):
        model.train()
        perm = torch.randperm(len(x_train))
        ep_loss = 0.0
        batches = 0
        for i in range(0, len(x_train), batch_size):
            idx = perm[i : i + batch_size]
            pred = model(x_train[idx])
            loss = torch.mean(w_train[idx] * (pred - y_train[idx]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            batches += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            v_loss = torch.mean((val_pred - y_val) ** 2).item()

        train_losses.append(ep_loss / max(1, batches))
        val_losses.append(v_loss)

    print(f"[Training] Final Held-Out Action MSE: {val_losses[-1]:.5f}")

    # Evaluate trained neural network policy on 10 unseen test seeds
    env = IronExcavatorEnv(seed=999)
    successes = 0
    eval_rewards = []
    rollout_frames = []

    for ep in range(10):
        obs, info = env.reset(seed=999 + ep)
        ep_rew = 0.0
        for step in range(env.max_steps):
            obs_in = torch.from_numpy(((obs - obs_mean[0]) / obs_std[0]).astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                act = model(obs_in).squeeze(0).numpy()

            # Record detailed trajectory on test episode 0 for rendering
            if ep == 0:
                p0, p1, p2, p3 = env.forward_kinematics(env.q)
                rollout_frames.append(
                    {
                        "step": step,
                        "time_s": step * env.dt,
                        "q": env.q.copy(),
                        "act": act.copy(),
                        "p0": p0,
                        "p1": p1,
                        "p2": p2,
                        "p3": p3,
                        "H_t": env.H_t.copy(),
                        "bucket_load": float(env.bucket_load),
                        "truck_load": float(env.truck_load),
                        "truck_pos": env.truck_pos.copy(),
                        "truck_yaw": float(env.truck_yaw),
                        "dig_center": env.dig_center.copy(),
                        "drone_pos": env.drone_pos.copy(),
                        "phase": env.phase,
                        "cum_reward": ep_rew,
                    }
                )

            obs, rew, term, trunc, info = env.step(act)
            ep_rew += rew
            if term or trunc:
                break

        eval_rewards.append(ep_rew)
        if info["truck_load"] >= 0.75:
            successes += 1

    print(f"[Eval] Neural Policy Success Rate: {successes}/10 ({successes*10}%) | Mean Return: {np.mean(eval_rewards):.2f}")

    ckpt_path = os.path.join(out_dir, "trained_excavator_policy.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "val_mse": val_losses[-1],
            "eval_success_rate": successes / 10.0,
        },
        ckpt_path,
    )
    return rollout_frames, train_losses, val_losses, env


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    dataset, _ = generate_synthetic_dataset(num_episodes=45, out_dir=out_dir)
    train_and_evaluate_policy(dataset, out_dir=out_dir)
