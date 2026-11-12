"""
Download one complete HDF5 trajectory file from fuxi-robot/excavator-motion
containing synchronized:
  - /observations/images/main      (Real Cab RGB Camera frames [N, 384, 480, 3])
  - /observations/images/elevation (Real LiDAR 2.5D Elevation Map [N, 200, 200, 3])
  - /observations/qpos             (Real 4-DOF Joint Angles [N, 4]: [boom, arm, bucket, swing])
  - /action                        (Real 4-DOF Target Actions [N, 4]: [boom, arm, bucket, swing])
"""

import os
import time
import urllib.request
import h5py
import numpy as np

DEST_H5 = "/Users/teja-san/Work/hcilab/iron_rl_gym/real_data/fuxi/xcmg75_real_trajectory.hdf5"
URL_H5 = "https://huggingface.co/datasets/fuxi-robot/excavator-motion/resolve/main/data/75/xcmg_data_2025-04-11-15-50-29.hdf5"


def main():
    if not os.path.exists(DEST_H5) or os.path.getsize(DEST_H5) < 500_000_000:
        t0 = time.time()
        print(f"Downloading real excavator HDF5 trajectory (~980 MB) from {URL_H5} ...")
        urllib.request.urlretrieve(URL_H5, DEST_H5)
        dt = time.time() - t0
        sz = os.path.getsize(DEST_H5) / 1e6
        print(f"Downloaded {sz:.1f} MB in {dt:.1f}s ({sz/max(0.1, dt):.1f} MB/s)")
    else:
        print(f"Using cached {DEST_H5} ({os.path.getsize(DEST_H5)/1e6:.1f} MB)")

    with h5py.File(DEST_H5, "r") as f:
        def show_tree(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name:35s} | shape={obj.shape} | dtype={obj.dtype}")
        print("\n=== HDF5 Structure ===")
        f.visititems(show_tree)


if __name__ == "__main__":
    main()
