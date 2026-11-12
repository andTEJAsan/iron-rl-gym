import os
import cv2
import h5py
import imageio
import argparse
import numpy as np
from dataclasses import dataclass


MAIN_BGR_H  = 384
MAIN_BGR_W  = 480
ELEVATION_H = 200
ELEVATION_W = 200


@dataclass
class ExcavatorModel:
    name: str = ''

    # body
    boom_len: float = 0.0
    arm_len: float = 0.0
    operating_arm_height: float = 0.0
    boom_init_angle: float = 0.0
    arm_init_angle: float = 0.0
    arm_offset_x: float = 0.0
    arm_offset_y: float = 0.0

    # bucket
    bucket_length: float = 0.0
    bucket_width: float = 0.0

    # elevation resolution
    meter_per_pixel_u: float = 0.0
    meter_per_pixel_v: float = 0.0


class ExcavatorKinematics(object):
    def __init__(self, excavator: str):
        self.model = self._get_model(excavator)

    def _get_model(self, excavator: str) -> ExcavatorModel:
        exc = ExcavatorModel(name=excavator)
        if excavator == '75':
            exc.boom_len = 3.6957
            exc.arm_len = 1.62233
            exc.operating_arm_height = 1.4
            exc.boom_init_angle = 1.0
            exc.arm_init_angle = 1.5
            exc.arm_offset_x = 0.0
            exc.arm_offset_y = -0.1
            exc.bucket_length = 0.8
            exc.bucket_width = 0.6
            exc.meter_per_pixel_u = 0.1
            exc.meter_per_pixel_v = 0.1
        
        elif excavator in ('490', '306'):
            exc.boom_len = 6.670
            exc.arm_len = 2.90746
            exc.operating_arm_height = 2.46
            exc.boom_init_angle = 1.0
            exc.arm_init_angle = 1.5
            exc.arm_offset_x = 0.0
            exc.arm_offset_y = 0.0
            exc.bucket_length = 1.5
            exc.bucket_width = 1.2
            exc.meter_per_pixel_u = 0.2
            exc.meter_per_pixel_v = 0.2
        
        else:
            raise NotImplementedError
        
        return exc

    def fk(self, joints: np.ndarray) -> np.ndarray:
        assert joints.shape[1] == 4

        boom = joints[:, 0]
        arm = joints[:, 1]
        x = self.model.boom_len * np.sin(boom + self.model.boom_init_angle) + \
            self.model.arm_len * np.sin(np.pi - boom - self.model.boom_init_angle - arm - self.model.arm_init_angle)
        z = self.model.boom_len * np.cos(boom + self.model.boom_init_angle) - \
            self.model.arm_len * np.cos(np.pi - boom - self.model.boom_init_angle - arm - self.model.arm_init_angle)
        y = np.zeros_like(x)
        xyz = np.stack([x, y, z], axis=1) 
        
        return xyz


def resize_with_aspect_ratio(
        image: np.ndarray,
        width: int=None,
        height: int=None
    ) -> np.ndarray:
    if width is None and height is None:
        return image
    
    h, w = image.shape[:2]
    ratio = min(height / h, width / w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    
    # keep ratio resize
    resized = cv2.resize(image, (new_w, new_h))
    
    # pad size
    pad_top = (height - new_h) // 2
    pad_bottom = height - new_h - pad_top
    pad_left = (width - new_w) // 2
    pad_right = width - new_w - pad_left
    
    # pad with zero
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded


def traj_visualization(
        excavator: str,
        data_path: str,
        out_dir: str,
        fmt: str='mp4',
        fps: int=30
    ) -> None:
    # load data
    with h5py.File(data_path) as f:
        elevations = np.array(f['observations']['images']['elevation'])
        joints = np.array(f['observations']['qpos'])
        mains = np.array(f['observations']['images']['main'])
    assert elevations.shape[0] == joints.shape[0]

    # forward kinemetics
    exc = ExcavatorKinematics(excavator)
    xyz = exc.fk(joints)
    px = (xyz[:, 0] / exc.model.meter_per_pixel_u).astype(np.int32)

    bucket_half_u = int(exc.model.bucket_length / 2 / exc.model.meter_per_pixel_u)
    bucket_half_v = int(exc.model.bucket_width / 2 / exc.model.meter_per_pixel_v)
    
    # visualization
    frames = []
    for i in range(elevations.shape[0]):
        # end pose
        traj_elevation = elevations[i].copy()
        center_u = ELEVATION_W // 2
        center_v = ELEVATION_H // 2 - px[i]
        traj_elevation[center_v - bucket_half_v: center_v + bucket_half_v,
                       center_u - bucket_half_u: center_u + bucket_half_u] = (255, 0, 0)

        # elevation
        traj_elevation = resize_with_aspect_ratio(traj_elevation, height=MAIN_BGR_H, width=MAIN_BGR_W)
        
        # main bgr
        traj_main = mains[i].copy()[..., ::-1]  # BGR -> RGB
        traj_main = resize_with_aspect_ratio(traj_main, height=MAIN_BGR_H, width=MAIN_BGR_W)
        traj_img = np.concatenate([traj_main, traj_elevation], axis=1)
        
        frames.append(traj_img)
    
    # save
    os.makedirs(out_dir, exist_ok=True)
    imageio.mimsave(f"{out_dir}/{data_path.split('/')[-1].split('.')[0]}.{fmt}", frames, fps=fps)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--excavator", type=str, required=True, help='excavator name')
    parser.add_argument("--data_path", type=str, required=True, help='path to the h5 file')
    parser.add_argument("--out_dir", type=str, required=True, help='path the output directory')
    parser.add_argument("--format", type=str, required=False, default='mp4', help='format of the output file, mp4 or gif are supported')
    parser.add_argument("--fps", type=int, required=False, default=30, help='FPS of the output file')
    args = parser.parse_args()

    traj_visualization(args.excavator, args.data_path, args.out_dir, args.format, args.fps)