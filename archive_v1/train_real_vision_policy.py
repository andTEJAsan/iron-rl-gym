"""
Real-World Vision-to-Action RL Pipeline on Real XCMG-75 Excavator Data
======================================================================
Trains a Dual-Stream Convolutional Vision-to-Action Policy (Cab RGB Camera +
2.5D LiDAR Elevation Map + Proprioception -> 4-DOF Joystick Commands & Target Joints)
using Advantage-Weighted Offline RL on 5 real-world Scoop-and-Dump cycles, and
evaluates on the held-out 6th real Scoop-and-Dump cycle.
"""

import os
import json
import cv2
import h5py
import imageio
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

H5_PATH = "/Users/teja-san/Work/hcilab/iron_rl_gym/real_data/fuxi/xcmg75_real_trajectory.hdf5"
OUT_DIR = "/Users/teja-san/Work/hcilab/iron_rl_gym/data"
os.makedirs(OUT_DIR, exist_ok=True)


def compute_real_fk_3d(joints: np.ndarray) -> np.ndarray:
    """
    Exact 3D forward kinematics for the XCMG-75 excavator (from fuxi-robot specs):
      joints: [N, 4] = [boom, arm, bucket, swing] in radians
      returns: [N, 3] = [X, Y, Z] bucket position in meters relative to cab pivot
    """
    boom_len = 3.6957
    arm_len = 1.62233
    boom_init = 1.0
    arm_init = 1.5

    boom = joints[:, 0]
    arm = joints[:, 1]
    bucket = joints[:, 2]
    swing = joints[:, 3]

    # Planar reach r and vertical height z
    r = (
        boom_len * np.sin(boom + boom_init)
        + arm_len * np.sin(np.pi - boom - boom_init - arm - arm_init)
        + 0.80 * np.sin(np.pi - boom - boom_init - arm - arm_init - 0.35 * bucket)
    )
    z = (
        boom_len * np.cos(boom + boom_init)
        - arm_len * np.cos(np.pi - boom - boom_init - arm - arm_init)
        - 0.80 * np.cos(np.pi - boom - boom_init - arm - arm_init - 0.35 * bucket)
    )
    x = r * np.cos(swing)
    y = r * np.sin(swing)
    return np.stack([x, y, z, r], axis=1)


def extract_real_rewards_and_phases(qpos, qvel, dig_pos, car_pos):
    """
    Compute dense step-by-step intermediate rewards R_t and operational phases
    from the real XCMG-75 sensor logs:
      qpos[:, 0]: Boom   | qpos[:, 1]: Arm/Stick
      qpos[:, 2]: Bucket | qpos[:, 3]: Swing (dig_pos[3]=-0.74, car_pos[3]=+0.61)
    """
    N = len(qpos)
    rewards = np.zeros(N, dtype=np.float32)
    r_dig_arr = np.zeros(N, dtype=np.float32)
    r_trans_arr = np.zeros(N, dtype=np.float32)
    r_dump_arr = np.zeros(N, dtype=np.float32)
    phases = []

    dig_swing = float(dig_pos[3])   # ~ -0.74 rad
    car_swing = float(car_pos[3])   # ~ +0.61 rad

    for t in range(N):
        boom, arm, bucket, swing = qpos[t]
        dboom, darm, dbucket, dswing = qvel[t]

        dist_to_dig = abs(swing - dig_swing)
        dist_to_car = abs(swing - car_swing)

        # 1. Digging reward: near trench (swing < -0.25) and curling bucket / pulling arm
        r_dig = 0.0
        if swing < -0.20:
            r_dig += 0.4 * max(0.0, 1.0 - dist_to_dig)
            if dbucket > 0.05:  # curling bucket to scoop soil
                r_dig += 2.5 * dbucket

        # 2. Transport reward: swinging from trench toward dump truck with curled bucket (bucket > 0.6)
        r_trans = 0.0
        if bucket > 0.55 and dswing > 0.03:
            r_trans = 3.0 * dswing * min(1.0, bucket / 1.5)

        # 3. Dump reward: over dump truck (dist_to_car < 0.35) and uncurling bucket (dbucket < -0.05)
        r_dump = 0.0
        if dist_to_car < 0.38 and dbucket < -0.05:
            r_dump = 4.0 * (-dbucket)

        # 4. Smoothness penalty
        p_smooth = 0.02 * float(np.sum(qvel[t] ** 2))

        rewards[t] = r_dig + r_trans + r_dump - p_smooth
        r_dig_arr[t] = r_dig
        r_trans_arr[t] = r_trans
        r_dump_arr[t] = r_dump

        # Phase tag
        if dist_to_car < 0.35 and (dbucket < -0.05 or bucket < 0.45):
            phases.append("DUMPING_IN_TRUCK")
        elif dswing > 0.06 and bucket > 0.50:
            phases.append("SWING_TO_TRUCK")
        elif dswing < -0.06 and swing > -0.45:
            phases.append("RETURN_TO_TRENCH")
        else:
            phases.append("DIGGING_SOIL")

    # Compute discounted Return-to-Go within each 15s cycle
    returns_to_go = np.zeros(N, dtype=np.float32)
    G = 0.0
    for t in reversed(range(N)):
        if t < N - 1 and phases[t] == "RETURN_TO_TRENCH" and phases[t + 1] == "DIGGING_SOIL":
            G = 0.0  # reset return boundary between scoop cycles
        G = rewards[t] + 0.96 * G
        returns_to_go[t] = G

    return rewards, r_dig_arr, r_trans_arr, r_dump_arr, returns_to_go, phases


class DualCameraVisionPolicy(nn.Module):
    """
    End-to-End Vision-to-Action Neural Network:
      Inputs:
        - cab_rgb:   [B, 3, 96, 128] Real Cab Camera RGB image
        - elev_rgb:  [B, 3, 96, 96]  Real Overhead 2.5D LiDAR Elevation image
        - proprio:   [B, 8]          Current 4 joint angles qpos + 4 previous velocities
      Outputs:
        - pred_joy:  [B, 4]          Normalized 4-DOF joystick velocity commands in [-1, 1]
        - pred_qnext:[B, 4]          Predicted next-step joint angles
    """

    def __init__(self):
        super().__init__()
        # 6-channel input: [cab_rgb(t), cab_rgb(t-2)] to perceive visual motion & bucket load from pixels
        self.cab_cnn = nn.Sequential(
            nn.Conv2d(6, 24, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.SiLU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((3, 4)),
            nn.Flatten(),
            nn.Linear(64 * 12, 128),
            nn.SiLU(),
        )

        self.elev_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(32 * 9, 64),
            nn.SiLU(),
        )

        # Proprioception is strictly the 4 joint angles qpos = [boom, arm, bucket, swing] (no velocity leak)
        self.proprio_mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(128 + 64 + 64, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Linear(192, 96),
            nn.SiLU(),
        )

        self.joy_out = nn.Sequential(nn.Linear(96, 4), nn.Tanh())
        self.delta_q_out = nn.Linear(96, 4)

    def forward(self, cab_rgb, elev_rgb, proprio):
        f_cab = self.cab_cnn(cab_rgb)
        f_elev = self.elev_cnn(elev_rgb)
        f_prop = self.proprio_mlp(proprio)
        fused = self.fusion_head(torch.cat([f_cab, f_elev, f_prop], dim=-1))
        joy = self.joy_out(fused)
        delta_q = self.delta_q_out(fused)
        return joy, delta_q


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Device] Training Vision-to-Action CNN on: {device}")

    with h5py.File(H5_PATH, "r") as f:
        mains_raw = f["observations/images/main"][:]          # [942, 480, 640, 3] BGR
        elevs_raw = f["observations/images/elevation"][:]     # [942, 200, 200, 3]
        qpos = f["observations/qpos"][:].astype(np.float32)   # [942, 4]
        qvel = f["observations/qvel"][:].astype(np.float32)   # [942, 4]
        action = f["action"][:].astype(np.float32)            # [942, 4]
        car_pos = f["observations/car_pos"][:].astype(np.float32)
        dig_pos = f["observations/dig_pos"][:].astype(np.float32)

    N = len(qpos)
    print(f"[Data] Loaded {N} real excavator steps (6 full Scoop-and-Dump cycles)")

    # Compute normalized joystick commands [-1, 1]^4 from joint velocities qvel
    qvel_max = np.percentile(np.abs(qvel), 98, axis=0, keepdims=True) + 1e-4
    joystick_gt = np.clip(qvel / qvel_max, -1.0, 1.0).astype(np.float32)
    delta_q_gt = (action - qpos).astype(np.float32)

    # Extract intermediate rewards and phases
    rewards, r_dig, r_trans, r_dump, returns_to_go, phases = extract_real_rewards_and_phases(
        qpos, qvel, dig_pos, car_pos
    )

    # Preprocess camera images to compact tensors: Cab [N, 3, 96, 128], Elev [N, 3, 96, 96]
    cab_single = np.zeros((N, 3, 96, 128), dtype=np.float32)
    elev_tensors = np.zeros((N, 3, 96, 96), dtype=np.float32)
    for i in range(N):
        rgb = cv2.cvtColor(mains_raw[i], cv2.COLOR_BGR2RGB)
        cab_small = cv2.resize(rgb, (128, 96), interpolation=cv2.INTER_AREA)
        cab_single[i] = np.transpose(cab_small, (2, 0, 1)) / 255.0

        el_small = cv2.resize(elevs_raw[i], (96, 96), interpolation=cv2.INTER_AREA)
        elev_tensors[i] = np.transpose(el_small, (2, 0, 1)) / 255.0

    # Stack frame t and frame t-2 for visual motion perception from pixels: [N, 6, 96, 128]
    cab_prev = np.concatenate([cab_single[:2], cab_single[:-2]], axis=0)
    cab_tensors = np.concatenate([cab_single, cab_prev], axis=1)

    # Proprioception input: strictly the 4 joint angles qpos(t) = [boom, arm, bucket, swing]
    proprio_raw = qpos.copy()
    prop_mean = proprio_raw.mean(axis=0, keepdims=True)
    prop_std = proprio_raw.std(axis=0, keepdims=True) + 1e-5
    proprio_norm = ((proprio_raw - prop_mean) / prop_std).astype(np.float32)

    # Advantage weights from real intermediate rewards
    adv = (returns_to_go - returns_to_go.mean()) / (returns_to_go.std() + 1e-5)
    weights = np.clip(np.exp(0.35 * adv), 0.4, 2.5).astype(np.float32)

    # Split: First 5 Scoop-and-Dump cycles (frames 0..760) = Train | Cycle 6 (frames 760..942) = Held-Out Test
    split = 760
    train_ds = TensorDataset(
        torch.from_numpy(cab_tensors[:split]),
        torch.from_numpy(elev_tensors[:split]),
        torch.from_numpy(proprio_norm[:split]),
        torch.from_numpy(joystick_gt[:split]),
        torch.from_numpy(delta_q_gt[:split]),
        torch.from_numpy(weights[:split]).unsqueeze(-1),
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    val_cab = torch.from_numpy(cab_tensors[split:]).to(device)
    val_elev = torch.from_numpy(elev_tensors[split:]).to(device)
    val_prop = torch.from_numpy(proprio_norm[split:]).to(device)
    val_joy_gt = torch.from_numpy(joystick_gt[split:]).to(device)

    model = DualCameraVisionPolicy().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=45)

    train_losses, val_losses = [], []
    for epoch in range(45):
        model.train()
        ep_loss = 0.0
        for b_cab, b_el, b_pr, b_joy, b_dq, b_w in train_loader:
            b_cab, b_el, b_pr = b_cab.to(device), b_el.to(device), b_pr.to(device)
            b_joy, b_dq, b_w = b_joy.to(device), b_dq.to(device), b_w.to(device)

            pred_joy, pred_dq = model(b_cab, b_el, b_pr)
            loss_joy = torch.mean(b_w * (pred_joy - b_joy) ** 2)
            loss_dq = torch.mean(b_w * (pred_dq - b_dq * 10.0) ** 2)
            loss = loss_joy + 0.5 * loss_dq

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss_joy.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            v_joy, _ = model(val_cab, val_elev, val_prop)
            v_mse = torch.mean((v_joy - val_joy_gt) ** 2).item()
        train_losses.append(ep_loss / len(train_loader))
        val_losses.append(v_mse)

    print(f"[Vision Policy] Held-Out Unseen Scoop Cycle (Cycle 6) Joystick MSE: {val_losses[-1]:.4f}")

    # Run inference across the entire trajectory & especially the held-out test cycle (760..942)
    model.eval()
    with torch.no_grad():
        all_pred_joy, all_pred_dq = [], []
        for i in range(0, N, 64):
            c_b = torch.from_numpy(cab_tensors[i : i + 64]).to(device)
            e_b = torch.from_numpy(elev_tensors[i : i + 64]).to(device)
            p_b = torch.from_numpy(proprio_norm[i : i + 64]).to(device)
            pj, pdq = model(c_b, e_b, p_b)
            all_pred_joy.append(pj.cpu().numpy())
            all_pred_dq.append((pdq.cpu().numpy()) / 10.0)
    pred_joy_np = np.concatenate(all_pred_joy, axis=0)
    pred_qpos_np = qpos + np.concatenate(all_pred_dq, axis=0)

    # Correlation on held-out test cycle
    corrs = [
        float(np.corrcoef(joystick_gt[split:, j], pred_joy_np[split:, j])[0, 1])
        for j in range(4)
    ]
    print(f"[Vision Policy] Held-Out Test Cycle Correlations [Boom, Arm, Bucket, Swing]: {[round(c, 3) for c in corrs]}")

    # Save trained real vision policy checkpoint
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "prop_mean": prop_mean,
            "prop_std": prop_std,
            "qvel_max": qvel_max,
            "test_mse": val_losses[-1],
            "test_correlations": corrs,
        },
        os.path.join(OUT_DIR, "real_vision_excavator_policy.pt"),
    )

    # --- Render 1: High-Res Real-World Vision Policy Evaluation Dashboard PNG ---
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16.5, 10.2), dpi=160)
    fig.patch.set_facecolor("#0b0f17")
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.12, 1.0], hspace=0.30, wspace=0.25)

    # Pick 4 key frames from the Held-Out Test Cycle (frames 760..942)
    test_frames = [790, 825, 862, 895]
    test_titles = [
        "1. DIGGING IN TRENCH (Real Cam + LiDAR)",
        "2. CURLING FULL BUCKET",
        "3. SWINGING TO DUMP TRUCK",
        "4. DUMPING INTO TRUCK BED",
    ]
    fk_3d_gt = compute_real_fk_3d(qpos)
    fk_3d_pred = compute_real_fk_3d(pred_qpos_np)

    for col, (fidx, stitle) in enumerate(zip(test_frames, test_titles)):
        ax_img = fig.add_subplot(gs[0, col])
        cab_rgb = cv2.cvtColor(mains_raw[fidx], cv2.COLOR_BGR2RGB)
        elev_rgb = elevs_raw[fidx].copy()

        # Draw bucket position marker on elevation map (green=GT, cyan=Vision Policy)
        r_gt = fk_3d_gt[fidx, 3]
        v_gt = int(np.clip(100 - r_gt / 0.1, 8, 192))
        cv2.circle(elev_rgb, (100, v_gt), 6, (0, 255, 100), -1)

        # Inset the 2.5D LiDAR elevation map in the top-right corner of the cab camera frame
        elev_inset = cv2.resize(elev_rgb, (175, 175))
        combined = cab_rgb.copy()
        combined[10:185, 640 - 185 : 640 - 10] = elev_inset
        cv2.rectangle(combined, (640 - 185, 10), (640 - 10, 185), (56, 189, 248), 2)

        ax_img.imshow(combined)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_title(
            f"{stitle}\nFrame {fidx} (t={fidx*0.1:.1f}s) | {phases[fidx]}",
            fontsize=9.2,
            fontweight="bold",
            color="#f8fafc",
        )

    # Bottom Panel 1: Held-Out Test Scoop (Frames 760..942) GT vs. Vision Policy Joystick Predictions
    ax_joy = fig.add_subplot(gs[1, 0:2])
    t_test = np.arange(split, N) * 0.1
    j_names = ["Boom", "Arm/Stick", "Bucket", "Swing"]
    j_cols = ["#facc15", "#4ade80", "#f472b6", "#38bdf8"]
    for j in range(4):
        ax_joy.plot(t_test, joystick_gt[split:, j], color=j_cols[j], lw=1.4, alpha=0.55, ls="--")
        ax_joy.plot(
            t_test,
            pred_joy_np[split:, j],
            color=j_cols[j],
            lw=2.1,
            label=f"{j_names[j]} (r={corrs[j]:.2f})",
        )
    ax_joy.axhline(0, color="#475569", lw=0.8)
    ax_joy.set_title(
        "Held-Out Unseen Real Scoop (Cycle 6): Human Operator (Dashed) vs. CNN Vision Policy (Solid)",
        fontsize=10,
        fontweight="bold",
        color="#f8fafc",
    )
    ax_joy.set_xlabel("Time in Recording (seconds)", fontsize=8.5)
    ax_joy.set_ylabel("Normalized Joystick Command [-1, 1]", fontsize=8.5)
    ax_joy.legend(loc="upper left", fontsize=8, ncol=4)
    ax_joy.grid(True, alpha=0.2)

    # Bottom Panel 2: Intermediate Reward Breakdown on Held-Out Test Scoop
    ax_rew = fig.add_subplot(gs[1, 2])
    ax_rew.fill_between(t_test, r_dig[split:], color="#f59e0b", alpha=0.45, label="R_dig (Scoop)")
    ax_rew.fill_between(t_test, r_trans[split:], color="#38bdf8", alpha=0.45, label="R_transport (Swing)")
    ax_rew.fill_between(t_test, r_dump[split:], color="#22c55e", alpha=0.45, label="R_dump (Truck Load)")
    ax_rew.plot(t_test, returns_to_go[split:] / 8.0, color="#f43f5e", lw=2.0, label="Return-to-Go G_t (÷8)")
    ax_rew.set_title("Real Intermediate Rewards (Unseen Cycle 6)", fontsize=10, fontweight="bold", color="#f8fafc")
    ax_rew.set_xlabel("Time (seconds)", fontsize=8.5)
    ax_rew.set_ylabel("Step Reward", fontsize=8.5)
    ax_rew.legend(loc="upper right", fontsize=7.5)
    ax_rew.grid(True, alpha=0.2)

    # Bottom Panel 3: Top-Down 2D Bucket Trajectory Across All 6 Real Scoops vs. Vision Policy
    ax_traj = fig.add_subplot(gs[1, 3])
    ax_traj.plot(fk_3d_gt[:split, 0], fk_3d_gt[:split, 1], color="#64748b", lw=1.0, alpha=0.5, label="Train Scoops 1–5 (GT)")
    ax_traj.plot(fk_3d_gt[split:, 0], fk_3d_gt[split:, 1], color="#22c55e", lw=2.5, ls="--", label="Test Scoop 6 (Human GT)")
    ax_traj.plot(fk_3d_pred[split:, 0], fk_3d_pred[split:, 1], color="#38bdf8", lw=2.2, label="Test Scoop 6 (Vision CNN)")
    ax_traj.scatter([fk_3d_gt[790, 0]], [fk_3d_gt[790, 1]], color="#f59e0b", s=85, zorder=10, label="Dig Trench Zone")
    ax_traj.scatter([fk_3d_gt[895, 0]], [fk_3d_gt[895, 1]], color="#f43f5e", s=85, marker="s", zorder=10, label="Dump Truck Bed")
    ax_traj.set_title("Bucket Tip Top-Down Arc (Meters)", fontsize=10, fontweight="bold", color="#f8fafc")
    ax_traj.set_xlabel("X Reach (m)", fontsize=8.5)
    ax_traj.set_ylabel("Y Lateral Swing (m)", fontsize=8.5)
    ax_traj.legend(loc="lower left", fontsize=7.2)
    ax_traj.grid(True, alpha=0.2)

    fig.suptitle(
        "Iron.inc Real-World Vision-to-Action RL Pipeline — Trained on Real XCMG-75 Excavator Video + 2.5D LiDAR",
        fontsize=14,
        fontweight="bold",
        color="#f8fafc",
        y=0.97,
    )

    dash_path = os.path.join(OUT_DIR, "real_excavator_vision_dashboard.png")
    fig.savefig(dash_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Render] Saved Real-World Vision Dashboard -> {dash_path}")

    # --- Render 2: Animated Real-Video HUD GIF & MP4 of Unseen Test Scoop (Frames 760..925) ---
    hud_frames = []
    for fidx in range(split, min(N, 926), 2):
        cab_rgb = cv2.cvtColor(mains_raw[fidx], cv2.COLOR_BGR2RGB)
        elev_rgb = elevs_raw[fidx].copy()

        # Draw bucket tip on LiDAR elevation map (Green = GT, Cyan = Vision CNN)
        v_gt = int(np.clip(100 - fk_3d_gt[fidx, 3] / 0.1, 8, 192))
        v_pr = int(np.clip(100 - fk_3d_pred[fidx, 3] / 0.1, 8, 192))
        cv2.circle(elev_rgb, (96, v_gt), 6, (34, 197, 94), -1)
        cv2.circle(elev_rgb, (104, v_pr), 6, (56, 189, 248), 2)

        elev_resized = cv2.resize(elev_rgb, (260, 260))

        # Build 920 x 480 HUD canvas
        canvas = np.zeros((480, 920, 3), dtype=np.uint8)
        canvas[:, :, :] = (15, 20, 28)
        canvas[:, :640] = cab_rgb

        # Top-right: 2.5D LiDAR Elevation Map
        canvas[12:272, 650:910] = elev_resized
        cv2.rectangle(canvas, (650, 12), (910, 272), (71, 85, 105), 1)
        cv2.putText(canvas, "2.5D LiDAR Elevation + Bucket FK", (655, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (226, 232, 240), 1, cv2.LINE_AA)

        # Bottom-right: Virtual Dual-Joystick HUD (Left Joystick: Swing/Arm, Right Joystick: Bucket/Boom)
        # Left box: (660, 310) to (765, 415) | Right box: (795, 310) to (900, 415)
        for bx, title, jx_idx, jy_idx in [
            (660, "Left Stick (Swing/Arm)", 3, 1),
            (795, "Right Stick (Bucket/Boom)", 2, 0),
        ]:
            by, bw = 310, 105
            cx, cy = bx + bw // 2, by + bw // 2
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bw), (30, 41, 59), -1)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bw), (100, 116, 139), 1)
            cv2.line(canvas, (bx, cy), (bx + bw, cy), (51, 65, 85), 1)
            cv2.line(canvas, (cx, by), (cx, by + bw), (51, 65, 85), 1)

            # GT Joystick (Green circle)
            gx = int(cx + joystick_gt[fidx, jx_idx] * 42)
            gy = int(cy - joystick_gt[fidx, jy_idx] * 42)
            cv2.circle(canvas, (gx, gy), 7, (34, 197, 94), -1)

            # Vision Policy Predicted Joystick (Cyan ring + line)
            px = int(cx + pred_joy_np[fidx, jx_idx] * 42)
            py = int(cy - pred_joy_np[fidx, jy_idx] * 42)
            cv2.line(canvas, (cx, cy), (px, py), (56, 189, 248), 2)
            cv2.circle(canvas, (px, py), 8, (56, 189, 248), 2)
            cv2.putText(canvas, title[:11], (bx + 4, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (203, 213, 225), 1, cv2.LINE_AA)

        # Legend & Live Phase/Reward Banner on Cab Video
        cv2.rectangle(canvas, (12, 12), (455, 82), (15, 20, 28), -1)
        cv2.rectangle(canvas, (12, 12), (455, 82), (56, 189, 248), 1)
        cv2.putText(
            canvas,
            f"UNSEEN TEST SCOOP | Frame {fidx} ({fidx*0.1:.1f}s)",
            (22, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (248, 250, 252),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Phase: {phases[fidx]} | Step Reward: +{rewards[fidx]:.2f}",
            (22, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (74, 222, 128),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Green = Human Operator   Cyan = Trained CNN Vision Policy",
            (652, 455),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (148, 163, 184),
            1,
            cv2.LINE_AA,
        )

        hud_frames.append(Image.fromarray(canvas))

    gif_path = os.path.join(OUT_DIR, "real_excavator_vision_rollout.gif")
    hud_frames[0].save(
        gif_path,
        save_all=True,
        append_images=hud_frames[1:],
        duration=110,
        loop=0,
    )
    print(f"[Render] Saved Real-Video Vision Policy HUD GIF ({len(hud_frames)} frames) -> {gif_path}")


if __name__ == "__main__":
    main()
