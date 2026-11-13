"""
Unified JCB 8026 Multimodal Policy & Critic Architecture (`models/policy.py`)
=============================================================================
Working backwards from the physical JCB 8026 deployment:
  - Input Observation Dict (identical on the JCB 8026 laptop and in JCB8026BenchEnv):
      * `proprio`:         float32[B, 12]       (4x WitMotion WT901 angles + 4x vel + 4x last stick)
      * `task_vec`:        float32[B, 8]        (Drone AprilTag trench & truck relative pose + fill)
      * `livox_elevation`: float32[B, 1, 32, 32](Upside-down cab-roof Livox Mid360S local heightmap)
      * `cab_rgb`:         float32[B, 3, 64, 64](Cab-roof Reolink Duo 2 PoE camera crop along boom)
  - Output Action Tensor:
      * `sticks`:          float32[B, 4] in [-1, +1]^4 @ 50 Hz
        Exact ISO Pattern (Section 3.3):
          idx 0 -> Ch 1: Left Fore-Aft  (Dipper / Stick)
          idx 1 -> Ch 2: Left Side      (Slew / Swing)
          idx 2 -> Ch 3: Right Fore-Aft (Boom)
          idx 3 -> Ch 4: Right Side     (Bucket Curl)
"""

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from deploy.jcb8026_config import CONFIG


class JCB8026Encoder(nn.Module):
    """Fuses WitMotion proprio + Drone AprilTag pose + Livox Mid360S heightmap + Reolink Cab RGB."""

    def __init__(self, latent_dim: int = 192):
        super().__init__()
        # Proprioception + AprilTag task vector branch (12 + 8 = 20D)
        self.state_mlp = nn.Sequential(
            nn.Linear(CONFIG.proprio_dim + CONFIG.task_vec_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 96),
            nn.SiLU(),
        )

        # C1: Livox Mid360S 1x32x32 local elevation map encoder
        self.livox_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # -> 16x16x16
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> 32x8x8
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)),                          # -> 32x2x2 = 128
            nn.Flatten(),
            nn.Linear(128, 48),
            nn.SiLU(),
        )

        # C3: Reolink Duo 2 PoE 3x64x64 cab-roof RGB encoder
        self.rgb_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),  # -> 16x32x32
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> 32x16x16
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)),                          # -> 32x2x2 = 128
            nn.Flatten(),
            nn.Linear(128, 48),
            nn.SiLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(96 + 48 + 48, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        state_in = torch.cat([obs["proprio"], obs["task_vec"]], dim=-1)
        z_state = self.state_mlp(state_in)
        z_livox = self.livox_cnn(obs["livox_elevation"])
        z_rgb = self.rgb_cnn(obs["cab_rgb"])
        return self.fusion(torch.cat([z_state, z_livox, z_rgb], dim=-1))


class JCB8026ActorCritic(nn.Module):
    """
    Unified Policy + Twin-Q / Value network used across:
      1. Stage 1 Behavior Cloning (`train/train_bc.py`)
      2. Stage 2 RL Fine-Tuning (`train/train_rl.py` - TD3+BC / PPO)
      3. 50 Hz Physical Deployment & Shadow Mode (`deploy/elrs_control_node.py`)
    """

    def __init__(self, latent_dim: int = 192):
        super().__init__()
        self.encoder = JCB8026Encoder(latent_dim=latent_dim)

        # Deterministic / Mean Actor head -> [-1, 1]^4 ISO joystick channels
        self.actor_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, CONFIG.action_dim),
            nn.Tanh(),
        )

        # Learnable log_std for stochastic RL exploration
        self.log_std = nn.Parameter(torch.full((CONFIG.action_dim,), -1.6))

        # Action-Value Critic Q(o, a) for TD3+BC / Advantage-Weighted RL fine-tuning
        self.q1_head = nn.Sequential(
            nn.Linear(latent_dim + CONFIG.action_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.q2_head = nn.Sequential(
            nn.Linear(latent_dim + CONFIG.action_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def act_deterministic(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.encoder(obs)
        return self.actor_head(z)

    def q_values(self, obs: Dict[str, torch.Tensor], action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(obs)
        za = torch.cat([z, action], dim=-1)
        return self.q1_head(za), self.q2_head(za)

    @torch.no_grad()
    def predict_sticks(self, obs_np: Dict[str, np.ndarray], device: str = "cpu") -> np.ndarray:
        """Single-step 50 Hz inference helper returning float32[4] in ISO channel order."""
        self.eval()
        obs_t = {
            k: torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
            for k, v in obs_np.items()
        }
        sticks = self.act_deterministic(obs_t).squeeze(0).cpu().numpy()
        return np.clip(sticks, -1.0, 1.0).astype(np.float32)
