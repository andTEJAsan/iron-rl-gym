"""
Render JCB 8026 Side-by-Side Closed-Loop Rollout Video (`benchmark/render_jcb8026_rollout_video.py`)
===================================================================================================
Visualizes the backwards-deployable `JCB8026ActorCritic` driving the canonical `JCB8026BenchEnv`
through `ELRSControlNode50Hz` in closed-loop:
  - Left Viewport:  Stage 1 Behavior Cloning (`jcb8026_bc_policy.pt`)
  - Right Viewport: Stage 2 RL Fine-Tuned (`jcb8026_rl_policy.pt`)
  - Insets:         Reolink Duo 2 PoE Cab Roof Camera (`cab_cam`) + DJI Mini 4 Pro Drone Camera (`drone_cam`)
  - Bottom HUD:     Live 4-Channel ISO Servo PWM (`1000..2000 us`), Deadman status, and 26-byte CRSF hex packet

Saves:
  - `data/jcb8026_il_vs_rl_rollout.mp4`
  - `data/jcb8026_il_vs_rl_rollout.gif`
"""

import os
import sys
import cv2
import mujoco
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from deploy.elrs_control_node import ControlMode, ELRSControlNode50Hz
from models.policy import JCB8026ActorCritic
from sim.jcb8026_env import JCB8026BenchEnv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints")
DATA_DIR = os.path.join(ROOT_DIR, "data")


def draw_pwm_bar(canvas: np.ndarray, x: int, y: int, w: int, h: int, pwm_us: int, label: str, color):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (45, 52, 62), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (95, 105, 118), 1)
    # Center line at 1500 us
    cx = x + w // 2
    cv2.line(canvas, (cx, y), (cx, y + h), (160, 165, 175), 1)

    frac = (float(pwm_us) - 1500.0) / 500.0  # -1..+1
    bar_px = int(frac * (w // 2 - 2))
    if bar_px >= 0:
        cv2.rectangle(canvas, (cx, y + 2), (cx + bar_px, y + h - 2), color, -1)
    else:
        cv2.rectangle(canvas, (cx + bar_px, y + 2), (cx, y + h - 2), color, -1)

    cv2.putText(
        canvas,
        f"{label}: {pwm_us:4d} us",
        (x + 6, y + h - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (240, 245, 250),
        1,
        cv2.LINE_AA,
    )


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[JCB 8026 Visualizer] Loading Stage 1 BC and Stage 2 RL checkpoints on {device}...")

    bc_policy = JCB8026ActorCritic().to(device)
    bc_policy.load_state_dict(torch.load(os.path.join(CKPT_DIR, "jcb8026_bc_policy.pt"), map_location=device))
    bc_policy.eval()

    rl_policy = JCB8026ActorCritic().to(device)
    rl_policy.load_state_dict(torch.load(os.path.join(CKPT_DIR, "jcb8026_rl_policy.pt"), map_location=device))
    rl_policy.eval()

    # Two identical JCB 8026 environments (same seed=1000) to compare Stage 1 BC vs Stage 2 RL head-to-head
    env_bc = JCB8026BenchEnv(scenario="T1_Nominal", seed=1000)
    env_rl = JCB8026BenchEnv(scenario="T1_Nominal", seed=1000)
    obs_bc, _ = env_bc.reset(seed=1000)
    obs_rl, _ = env_rl.reset(seed=1000)

    node_bc = ELRSControlNode50Hz(policy=bc_policy, mode=ControlMode.AUTONOMOUS, device=device)
    node_rl = ELRSControlNode50Hz(policy=rl_policy, mode=ControlMode.AUTONOMOUS, device=device)

    # Offscreen OpenGL renderers
    rend_bc = mujoco.Renderer(env_bc.model, height=420, width=620)
    rend_rl = mujoco.Renderer(env_rl.model, height=420, width=620)
    rend_inset = mujoco.Renderer(env_rl.model, height=130, width=190)

    out_w, out_h = 1280, 680
    mp4_path = os.path.join(DATA_DIR, "jcb8026_il_vs_rl_rollout.mp4")
    gif_path = os.path.join(DATA_DIR, "jcb8026_il_vs_rl_rollout.gif")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(mp4_path, fourcc, 25.0, (out_w, out_h))
    gif_frames = []

    total_steps = env_rl.max_steps
    for step in range(total_steps):
        now = step * env_rl.dt
        pkt_bc = env_bc._build_sensor_packet()
        pkt_rl = env_rl._build_sensor_packet()

        rec_bc = node_bc.step_50hz(pkt_bc, xbox_sticks=np.zeros(4), deadman_held=True, now=now)
        rec_rl = node_rl.step_50hz(pkt_rl, xbox_sticks=np.zeros(4), deadman_held=True, now=now)

        obs_bc, _, _, _, info_bc = env_bc.step(rec_bc["active_sticks"])
        obs_rl, _, _, _, info_rl = env_rl.step(rec_rl["active_sticks"])

        # Render every 2nd step for smooth video + compact GIF
        if step % 2 == 0 or step == total_steps - 1:
            canvas = np.full((out_h, out_w, 3), (15, 19, 26), dtype=np.uint8)

            # Top Title Banner
            cv2.rectangle(canvas, (0, 0), (out_w, 46), (24, 30, 40), -1)
            cv2.putText(
                canvas,
                f"JCB 8026 CTS Closed-Loop Deployment Sim | t = {now:4.2f}s / {env_rl.horizon_sec:.1f}s | Phase: {info_rl['phase']}",
                (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (235, 240, 248),
                2,
                cv2.LINE_AA,
            )

            # Left Viewport: Stage 1 BC
            rend_bc.update_scene(env_bc.data, camera="cinema_cam")
            img_bc = cv2.cvtColor(rend_bc.render(), cv2.COLOR_RGB2BGR)
            canvas[54 : 54 + 420, 14 : 14 + 620] = img_bc
            cv2.rectangle(canvas, (14, 54), (14 + 620, 54 + 420), (88, 166, 255), 2)
            cv2.rectangle(canvas, (14, 54), (430, 90), (20, 26, 36), -1)
            cv2.putText(
                canvas,
                f"STAGE 1: IMITATION (BC) | Bucket: {info_bc['in_bucket']:2d} | Truck: {info_bc['in_truck']:2d}",
                (22, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (110, 190, 255),
                1,
                cv2.LINE_AA,
            )

            # Right Viewport: Stage 2 RL Fine-Tuned
            rend_rl.update_scene(env_rl.data, camera="cinema_cam")
            img_rl = cv2.cvtColor(rend_rl.render(), cv2.COLOR_RGB2BGR)
            canvas[54 : 54 + 420, 646 : 646 + 620] = img_rl
            cv2.rectangle(canvas, (646, 54), (646 + 620, 54 + 420), (63, 185, 80), 2)
            cv2.rectangle(canvas, (646, 54), (1080, 90), (20, 26, 36), -1)
            cv2.putText(
                canvas,
                f"STAGE 2: RL FINE-TUNED  | Bucket: {info_rl['in_bucket']:2d} | Truck: {info_rl['in_truck']:2d}",
                (654, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (90, 230, 115),
                1,
                cv2.LINE_AA,
            )

            # Sensor Insets on Right Viewport (C3 Reolink Cab Cam + A1 DJI Mini 4 Pro Drone Cam)
            rend_inset.update_scene(env_rl.data, camera="cab_cam")
            cab_bgr = cv2.cvtColor(rend_inset.render(), cv2.COLOR_RGB2BGR)
            canvas[336 : 336 + 130, 1068 : 1068 + 190] = cab_bgr
            cv2.rectangle(canvas, (1068, 336), (1068 + 190, 336 + 130), (220, 225, 230), 1)
            cv2.putText(canvas, "C3: Reolink Cab PoE", (1074, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

            rend_inset.update_scene(env_rl.data, camera="drone_cam")
            drone_bgr = cv2.cvtColor(rend_inset.render(), cv2.COLOR_RGB2BGR)
            canvas[198 : 198 + 130, 1068 : 1068 + 190] = drone_bgr
            cv2.rectangle(canvas, (1068, 198), (1068 + 190, 198 + 130), (220, 225, 230), 1)
            cv2.putText(canvas, "A1: DJI Mini 4 Pro", (1074, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

            # Bottom Telemetry Panel (50 Hz ELRS Servo PWM Bars + CRSF Frame)
            cv2.rectangle(canvas, (14, 484), (1266, 666), (22, 28, 36), -1)
            cv2.rectangle(canvas, (14, 484), (1266, 666), (60, 70, 82), 1)

            pwm_bc = rec_bc["pwm_us"]
            pwm_rl = rec_rl["pwm_us"]
            ch_labels = [
                "Ch1 Left Fore-Aft (Dipper)",
                "Ch2 Left Side (Slew)",
                "Ch3 Right Fore-Aft (Boom)",
                "Ch4 Right Side (Bucket)",
            ]
            ch_colors = [(255, 170, 80), (80, 165, 255), (100, 220, 110), (235, 145, 255)]

            cv2.putText(canvas, "Stage 1 BC Servo PWM (ER5C V2):", (28, 508), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (170, 200, 240), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Stage 2 RL Servo PWM (ER5C V2) + Live CRSF Serial Stream:", (654, 508), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 235, 160), 1, cv2.LINE_AA)

            for i in range(4):
                draw_pwm_bar(canvas, 28, 518 + i * 28, 580, 22, int(pwm_bc[i]), ch_labels[i], ch_colors[i])
                draw_pwm_bar(canvas, 654, 518 + i * 28, 580, 22, int(pwm_rl[i]), ch_labels[i], ch_colors[i])

            q_iso = env_rl.get_iso_qpos()
            cv2.putText(
                canvas,
                f"C4 WitMotion WT901 (rad): Dipper={q_iso[0]:+.2f} Slew={q_iso[1]:+.2f} Boom={q_iso[2]:+.2f} Bucket={q_iso[3]:+.2f}",
                (28, 652),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (205, 212, 222),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"B4/B5 FTDI->ELRS CRSF (26B): 0x{rec_rl['crsf_hex'][:36]}... | Deadman: HELD (SAFE)",
                (654, 652),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (120, 235, 145),
                1,
                cv2.LINE_AA,
            )

            writer.write(canvas)
            if step % 4 == 0 or step == total_steps - 1:
                rgb_frame = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                small_pil = Image.fromarray(rgb_frame).resize((800, 425), Image.Resampling.LANCZOS)
                gif_frames.append(small_pil)

    writer.release()
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=85,
        loop=0,
    )
    print(f"[JCB 8026 Visualizer] Saved MP4 to {mp4_path} and GIF to {gif_path}")
    print(f"  Final Truck Boulders -> Stage 1 BC: {info_bc['in_truck']}, Stage 2 RL: {info_rl['in_truck']}")


if __name__ == "__main__":
    main()
