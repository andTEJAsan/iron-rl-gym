"""
Render visual preview (High-Res Dashboard PNG + Animated Rollout GIF)
for the Iron.inc Physical RL Gym & Synthetic Data Pipeline.
"""

import os
import numpy as np

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from excavator_gym import (
    generate_synthetic_dataset,
    train_and_evaluate_policy,
)


def draw_3d_scene(ax, frame, env, title_prefix=""):
    ax.cla()
    ax.set_facecolor("#0f141c")

    # 1. Plot 2.5D terrain surface H_t(x, y)
    X, Y = env.X, env.Y
    H = frame["H_t"]
    ax.plot_surface(
        X, Y, H,
        cmap="copper",
        alpha=0.78,
        linewidth=0,
        antialiased=True,
        vmin=-0.2,
        vmax=0.85,
    )

    # 2. Draw Excavator Base & Cab at origin
    p0, p1, p2, p3 = frame["p0"], frame["p1"], frame["p2"], frame["p3"]
    ax.plot([0, 0], [0, 0], [0, p0[2]], color="#eab308", lw=7, solid_capstyle="round")
    ax.scatter([0], [0], [p0[2]], color="#facc15", s=110, edgecolors="black", zorder=10)

    # 3. Draw Articulated Boom -> Stick -> Bucket Links
    xs = [p0[0], p1[0], p2[0], p3[0]]
    ys = [p0[1], p1[1], p2[1], p3[1]]
    zs = [p0[2], p1[2], p2[2], p3[2]]
    ax.plot(xs, ys, zs, color="#f59e0b", lw=5.5, marker="o", markersize=6, markerfacecolor="#1f2937", markeredgecolor="#facc15")

    # Bucket soil load sphere at p3
    bucket_color = "#a16207" if frame["bucket_load"] > 0.2 else "#38bdf8"
    bucket_size = 55 + 160 * frame["bucket_load"]
    ax.scatter([p3[0]], [p3[1]], [p3[2]], color=bucket_color, s=bucket_size, edgecolors="white", lw=1.2, zorder=15)

    # 4. Draw Pickup Truck Cargo Bed Box
    tx, ty, tz = frame["truck_pos"]
    yaw = frame["truck_yaw"]
    L, W, H_bed = 1.35, 0.90, 0.55
    corners_local = np.array([
        [-L/2, -W/2], [L/2, -W/2], [L/2, W/2], [-L/2, W/2], [-L/2, -W/2]
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners_xy = corners_local @ R.T + np.array([tx, ty])

    # Bottom and top rims of pickup truck bed
    ax.plot(corners_xy[:, 0], corners_xy[:, 1], np.full(5, 0.22), color="#38bdf8", lw=2.0)
    ax.plot(corners_xy[:, 0], corners_xy[:, 1], np.full(5, tz), color="#38bdf8", lw=2.8)
    for i in range(4):
        ax.plot([corners_xy[i, 0]]*2, [corners_xy[i, 1]]*2, [0.22, tz], color="#38bdf8", lw=1.8)

    # Draw soil fill level inside truck bed
    if frame["truck_load"] > 0.05:
        fill_z = 0.25 + 0.48 * min(1.0, frame["truck_load"])
        ax.scatter([tx], [ty], [fill_z], color="#b45309", s=60 + 180 * frame["truck_load"], alpha=0.9)

    # 5. Draw Overhead Autonomous Drone + LiDAR/Depth Scan Cone
    dx, dy, dz = frame["drone_pos"]
    ax.scatter([dx], [dy], [dz], color="#22d3ee", s=70, marker="^", edgecolors="white", lw=1.2)
    # Scan rays to dig center and truck
    dig_x, dig_y = frame["dig_center"]
    ax.plot([dx, dig_x], [dy, dig_y], [dz, 0.2], color="#22d3ee", ls="--", lw=0.9, alpha=0.55)
    ax.plot([dx, tx], [dy, ty], [dz, tz], color="#4ade80", ls="--", lw=0.9, alpha=0.55)

    ax.set_xlim(-4.2, 4.5)
    ax.set_ylim(-4.2, 4.5)
    ax.set_zlim(-0.3, 6.5)
    ax.view_init(elev=25, azim=-52)

    ax.tick_params(colors="#94a3b8", labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.18)
    ax.set_title(
        f"{title_prefix}t={frame['time_s']:.1f}s | Phase: {frame['phase']}",
        color="#f8fafc",
        fontsize=10,
        fontweight="bold",
        pad=6,
    )


def create_dashboard_and_gif(rollout_frames, train_losses, val_losses, env, out_dir):
    plt.style.use("dark_background")

    # --- 1. High-Res Summary Dashboard PNG ---
    fig = plt.figure(figsize=(16, 10), dpi=160)
    fig.patch.set_facecolor("#0b0f17")
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.25, 1.0], hspace=0.30, wspace=0.26)

    # Pick 4 key frames across the rollout corresponding to the 4 physical phases
    def find_phase_idx(phase_name, fallback_frac):
        matches = [i for i, f in enumerate(rollout_frames) if f["phase"] == phase_name]
        if matches:
            return matches[len(matches) // 2]
        return min(len(rollout_frames) - 1, int(len(rollout_frames) * fallback_frac))

    key_indices = [
        find_phase_idx("DIGGING", 0.18),
        find_phase_idx("SWING_TO_TRUCK", 0.45),
        find_phase_idx("DUMPING", 0.70),
        find_phase_idx("TRUCK_DEPART", 0.92),
    ]
    stage_titles = [
        "1. DIG & SCOOP\n",
        "2. SWING TO TRUCK\n",
        "3. DUMP IN BED\n",
        "4. TRUCK DEPARTS\n",
    ]

    for col, (idx, st_title) in enumerate(zip(key_indices, stage_titles)):
        ax3d = fig.add_subplot(gs[0, col], projection="3d")
        draw_3d_scene(ax3d, rollout_frames[idx], env, title_prefix=st_title)

    # Bottom Row Panel 1: Drone 2.5D Heightmap Diff (Delta H)
    ax_map = fig.add_subplot(gs[1, 0])
    H_diff = rollout_frames[-1]["H_t"] - rollout_frames[0]["H_t"]
    im = ax_map.imshow(
        H_diff,
        extent=[-env.domain_lim, env.domain_lim, -env.domain_lim, env.domain_lim],
        origin="lower",
        cmap="coolwarm",
        vmin=-0.55,
        vmax=0.55,
    )
    ax_map.scatter([0], [0], color="#facc15", s=60, label="JCB Base")
    ax_map.scatter(
        [rollout_frames[0]["dig_center"][0]],
        [rollout_frames[0]["dig_center"][1]],
        color="#38bdf8",
        marker="x",
        s=70,
        label="Excavated Trench (-ΔH)",
    )
    ax_map.scatter(
        [rollout_frames[0]["truck_pos"][0]],
        [rollout_frames[0]["truck_pos"][1]],
        color="#f87171",
        marker="s",
        s=55,
        label="Loaded Truck (+ΔH)",
    )
    ax_map.set_title("Drone Overhead Heightmap Diff (ΔH)", fontsize=10, fontweight="bold", color="#f8fafc")
    ax_map.set_xlabel("X (meters)", fontsize=8)
    ax_map.set_ylabel("Y (meters)", fontsize=8)
    ax_map.legend(loc="lower left", fontsize=7, framealpha=0.6)
    plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)

    # Bottom Row Panel 2: Neural Policy Continuous Joystick Actions a_t
    ax_act = fig.add_subplot(gs[1, 1])
    times = [f["time_s"] for f in rollout_frames]
    acts = np.array([f["act"] for f in rollout_frames])
    labels = ["Swing (a₁)", "Boom (a₂)", "Stick (a₃)", "Bucket (a₄)"]
    colors = ["#38bdf8", "#facc15", "#4ade80", "#f472b6"]
    for i in range(4):
        ax_act.plot(times, acts[:, i], label=labels[i], color=colors[i], lw=1.8)
    ax_act.axhline(0, color="#475569", lw=0.8, ls="--")
    ax_act.set_ylim(-1.15, 1.15)
    ax_act.set_title("Neural Policy Joystick Commands (20 Hz)", fontsize=10, fontweight="bold", color="#f8fafc")
    ax_act.set_xlabel("Time (s)", fontsize=8)
    ax_act.set_ylabel("Command [-1, 1]", fontsize=8)
    ax_act.legend(loc="upper right", fontsize=7, ncol=2)
    ax_act.grid(True, alpha=0.2)

    # Bottom Row Panel 3: Soil Mass Transfer & Cumulative RL Reward
    ax_load = fig.add_subplot(gs[1, 2])
    b_load = [f["bucket_load"] * 100 for f in rollout_frames]
    t_load = [min(1.0, f["truck_load"]) * 100 for f in rollout_frames]
    ax_load.fill_between(times, b_load, color="#f59e0b", alpha=0.35, label="Bucket Fill (%)")
    ax_load.plot(times, b_load, color="#f59e0b", lw=2.2)
    ax_load.fill_between(times, t_load, color="#22c55e", alpha=0.35, label="Truck Bed Fill (%)")
    ax_load.plot(times, t_load, color="#22c55e", lw=2.2)
    ax_load.set_ylim(0, 110)
    ax_load.set_title("Soil Volume Transfer (Bucket → Truck)", fontsize=10, fontweight="bold", color="#f8fafc")
    ax_load.set_xlabel("Time (s)", fontsize=8)
    ax_load.set_ylabel("Capacity (%)", fontsize=8)
    ax_load.legend(loc="center right", fontsize=8)
    ax_load.grid(True, alpha=0.2)

    # Bottom Row Panel 4: Offline RL / Policy Training Convergence & Cumulative Return
    ax_tr = fig.add_subplot(gs[1, 3])
    epochs = np.arange(1, len(train_losses) + 1)
    ax_tr.plot(epochs, train_losses, color="#38bdf8", lw=2.0, label="Train Loss (AWR)")
    ax_tr.plot(epochs, val_losses, color="#f43f5e", lw=2.0, ls="--", label="Held-Out Val MSE")
    ax_tr.set_yscale("log")
    ax_tr.set_title("Policy Training Curve (45 Synthetic Eps)", fontsize=10, fontweight="bold", color="#f8fafc")
    ax_tr.set_xlabel("Epoch", fontsize=8)
    ax_tr.set_ylabel("Action MSE (log scale)", fontsize=8)
    ax_tr.legend(loc="upper right", fontsize=8)
    ax_tr.grid(True, alpha=0.2)

    fig.suptitle(
        "Iron.inc Physical RL Gym — Autonomous Excavator + Drone + Haul Truck Rollout (Trained Policy)",
        fontsize=14,
        fontweight="bold",
        color="#f8fafc",
        y=0.97,
    )

    dashboard_path = os.path.join(out_dir, "iron_rl_gym_preview.png")
    fig.savefig(dashboard_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Render] Saved dashboard preview -> {dashboard_path}")

    # --- 2. Animated Rollout GIF ---
    gif_frames = []
    fig_anim = plt.figure(figsize=(10, 5.2), dpi=110)
    fig_anim.patch.set_facecolor("#0b0f17")
    gs_anim = GridSpec(1, 2, figure=fig_anim, width_ratios=[1.35, 1.0], wspace=0.25)
    ax3d_anim = fig_anim.add_subplot(gs_anim[0, 0], projection="3d")
    ax_tel = fig_anim.add_subplot(gs_anim[0, 1])

    # Render every 2nd step for a smooth ~50-frame GIF
    for idx in range(0, len(rollout_frames), 2):
        f = rollout_frames[idx]
        draw_3d_scene(ax3d_anim, f, env, title_prefix="Trained Policy | ")

        ax_tel.cla()
        ax_tel.set_facecolor("#0f141c")
        t_slice = [rf["time_s"] for rf in rollout_frames[: idx + 1]]
        b_slice = [rf["bucket_load"] * 100 for rf in rollout_frames[: idx + 1]]
        tr_slice = [min(1.0, rf["truck_load"]) * 100 for rf in rollout_frames[: idx + 1]]
        rew_slice = [rf["cum_reward"] for rf in rollout_frames[: idx + 1]]

        ax_tel.plot(t_slice, b_slice, color="#f59e0b", lw=2.2, label="Bucket Load (%)")
        ax_tel.plot(t_slice, tr_slice, color="#22c55e", lw=2.2, label="Truck Bed Load (%)")
        ax_tel.plot(t_slice, rew_slice, color="#38bdf8", lw=2.0, ls="--", label="Cumulative Reward")
        ax_tel.set_xlim(0, rollout_frames[-1]["time_s"] + 0.2)
        ax_tel.set_ylim(-5, 115)
        ax_tel.set_title("Live Telemetry & RL Reward", fontsize=10, fontweight="bold", color="#f8fafc")
        ax_tel.set_xlabel("Time (s)", fontsize=8)
        ax_tel.legend(loc="upper left", fontsize=8)
        ax_tel.grid(True, alpha=0.2)

        fig_anim.canvas.draw()
        rgba = np.asarray(fig_anim.canvas.buffer_rgba())
        gif_frames.append(Image.fromarray(rgba[:, :, :3]))

    plt.close(fig_anim)
    gif_path = os.path.join(out_dir, "iron_rl_rollout.gif")
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=90,
        loop=0,
    )
    print(f"[Render] Saved animated rollout GIF ({len(gif_frames)} frames) -> {gif_path}")
    return dashboard_path, gif_path


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    dataset, _ = generate_synthetic_dataset(num_episodes=45, out_dir=data_dir)
    rollout_frames, train_losses, val_losses, env = train_and_evaluate_policy(dataset, out_dir=data_dir)
    create_dashboard_and_gif(rollout_frames, train_losses, val_losses, env, out_dir=data_dir)
