"""
Download real-world excavator datasets (video + synchronized joystick/joint telemetry)
from Hugging Face:
  1. FlywheelAI/excavator-dataset (Multi-view excavator video + CSV joystick logs)
  2. fuxi-robot/excavator-motion & fuxi-robot/excavator-video (Truck-loading RGB + LiDAR elevation + 4-DOF joint actions)
"""

import os
import json
import time
import urllib.request

BASE_DIR = "/Users/teja-san/Work/hcilab/iron_rl_gym/real_data"
os.makedirs(f"{BASE_DIR}/flywheel", exist_ok=True)
os.makedirs(f"{BASE_DIR}/fuxi", exist_ok=True)


def fetch_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())


def download_file(url: str, dest: str):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"  [Cached] {dest} ({os.path.getsize(dest)/1e6:.2f} MB)")
        return
    t0 = time.time()
    print(f"  [Downloading] {url} -> {dest} ...")
    urllib.request.urlretrieve(url, dest)
    dt = time.time() - t0
    sz = os.path.getsize(dest) / 1e6
    print(f"  [Done] {sz:.2f} MB in {dt:.1f}s ({sz/max(0.1, dt):.1f} MB/s)")


def main():
    # 1. Download FlywheelAI/excavator-dataset session 20250828132109
    print("=== 1. Downloading FlywheelAI Real Excavator Video + Joystick Telemetry ===")
    fw_base = "https://huggingface.co/datasets/FlywheelAI/excavator-dataset/resolve/main/20250828132109"
    download_file(f"{fw_base}/D05_20250828132109.csv", f"{BASE_DIR}/flywheel/D05_joystick.csv")
    download_file(f"{fw_base}/D01_20250828132109.mp4", f"{BASE_DIR}/flywheel/D01_front_cam.mp4")
    download_file(f"{fw_base}/D02_20250828132109.mp4", f"{BASE_DIR}/flywheel/D02_side_cam.mp4")

    # 2. Inspect and download fuxi-robot/excavator-motion assets & fuxi-robot/excavator-video clip
    print("\n=== 2. Inspecting fuxi-robot/excavator-motion & excavator-video ===")
    try:
        assets = fetch_json("https://huggingface.co/api/datasets/fuxi-robot/excavator-motion/tree/main/assets")
        for it in assets:
            p = it.get("path")
            sz = it.get("size", 0)
            print(f"  fuxi-motion asset: {p} ({sz/1e6:.2f} MB)")
            if sz > 0 and sz < 100_000_000:
                fname = os.path.basename(p)
                download_file(
                    f"https://huggingface.co/datasets/fuxi-robot/excavator-motion/resolve/main/{p}",
                    f"{BASE_DIR}/fuxi/{fname}",
                )
    except Exception as e:
        print("  fuxi-motion assets check error:", e)

    try:
        v301 = fetch_json("https://huggingface.co/api/datasets/fuxi-robot/excavator-video/tree/main/videos/301")
        for it in v301[:5]:
            p = it.get("path")
            sz = it.get("size", 0)
            print(f"  fuxi-video/301: {p} ({sz/1e6:.2f} MB)")
            if sz > 0 and sz < 80_000_000:
                fname = os.path.basename(p)
                download_file(
                    f"https://huggingface.co/datasets/fuxi-robot/excavator-video/resolve/main/{p}",
                    f"{BASE_DIR}/fuxi/{fname}",
                )
                break
    except Exception as e:
        print("  fuxi-video check error:", e)

    # Also download data.yaml from fuxi-robot/excavator-video (annotations of real truck loading clips!)
    try:
        download_file(
            "https://huggingface.co/datasets/fuxi-robot/excavator-video/resolve/main/data.yaml",
            f"{BASE_DIR}/fuxi/data.yaml",
        )
    except Exception as e:
        print("  data.yaml error:", e)

    print("\nAll downloads complete!")


if __name__ == "__main__":
    main()
