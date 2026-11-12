import os
import h5py
import shutil
import argparse
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


REPO_NAME = "your_hf_username/excavator-motion"


def run(data_dir: str, out_dir: str):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    # Create LeRobot dataset, define features to store
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        fps=10,
        root=out_dir,
        robot_type="excavator",
        features={
            "main": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "elevation": {
                "dtype": "image",
                "shape": (200, 200, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (4,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (4,),
                "names": ["action"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over excavator-motition datasets and write episodes to the LeRobot dataset
    data_list = [f'{data_dir}/{name}' for name in os.listdir(data_dir)]
    for data_path in data_list:
        with h5py.File(data_path) as f:
            mains = np.array(f['observations']['images']['main'])
            elevations = np.array(f['observations']['images']['elevation'])
            states = np.array(f['observations']['qpos'])
            actions = np.array(f['action'])

            n_frames = actions.shape[0]
            for i in range(n_frames):
                dataset.add_frame(frame={
                        "main": mains[i],
                        "elevation": elevations[i],
                        "state": states[i],
                        "action": actions[i]
                    },
                    task="",
                )
            dataset.save_episode()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help='path of data to be converted')
    parser.add_argument("--out_dir", type=str, required=True, help='path for saving conversion results')
    args = parser.parse_args()

    run(args.data_dir, args.out_dir)