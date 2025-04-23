import os

from sympy import false

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import torch

ALBEDO = 'albedo'
COLOR = 'color'
SHADING_NORMAL = 'shading_normal'
WORLD_POSITION = 'world_position'
NORMALIZED_WORLD_POSITION = 'normalized_world_position'
REFERENCE = 'reference'
FEATURES = 'features'
FILE_EXT = '.exr'
INPUT = 'input'
TARGET = 'target'

samples_per_scene = 60

def load_exr(file_path: str) -> torch.Tensor:
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    float_data = img.astype(np.float32)
    tensor = torch.from_numpy(float_data).permute(2, 0, 1)

    return tensor


def save_exr(output_tensor: torch.Tensor, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 转换为 numpy 数组并调整维度顺序 (C, H, W) -> (H, W, C)
    np_data = output_tensor.detach().cpu().numpy().transpose(1, 2, 0)

    if np_data.dtype != np.float32:
        np_data = np_data.astype(np.float32)

    success = cv2.imwrite(save_path, np_data)

    if not success:
        raise RuntimeError(f"Failed to save EXR file: {save_path}")


def load_single_sample(data_path: str, idx: int) -> dict:
    def build_path(channel_name):
        return f"{data_path}/{channel_name}{idx}{FILE_EXT}"

    color = load_exr(build_path(COLOR))

    albedo = load_exr(build_path(ALBEDO))
    shading_normal = load_exr(build_path(SHADING_NORMAL))
    world_position = load_exr(build_path(WORLD_POSITION))

    reference = load_exr(build_path(REFERENCE))

    sample = {
        COLOR: color,
        ALBEDO: albedo,
        SHADING_NORMAL: shading_normal,
        WORLD_POSITION: world_position,
        REFERENCE: reference
    }

    return sample


def load_all_normalized_samples(data_path: str) -> list[dict]:
    normalized_samples = []

    for idx in range(samples_per_scene):
        sample = load_single_sample(data_path, idx)
        color = sample[COLOR]
        albedo = sample[ALBEDO]
        shading_normal = sample[SHADING_NORMAL]
        world_position = sample[WORLD_POSITION]
        reference = sample[REFERENCE]

        image_with_noise = color * albedo

        pos_mean = world_position.mean()
        pos_std = world_position.std()
        norm_pos = (world_position - pos_mean) / (pos_std + 1e-6)

        normalized_sample = {
            INPUT: image_with_noise,
            TARGET: reference,

            ALBEDO: albedo,
            SHADING_NORMAL: shading_normal,
            NORMALIZED_WORLD_POSITION: norm_pos,
        }

        normalized_samples.append(normalized_sample)

    return normalized_samples


if __name__ == "__main__":
    path = "../dataset/classroom/inputs"
    samples = load_all_normalized_samples(path)

    global_min = float("inf")  # 记录所有样本的最小值
    global_max = float("-inf")  # 记录所有样本的最大值

    for sample in samples:
        pos = sample[NORMALIZED_WORLD_POSITION]

        min_val = pos.min().item()
        max_val = pos.max().item()

        # 更新全局最大最小值
        global_min = min(global_min, min_val)
        global_max = max(global_max, max_val)

    # 输出数据范围
    print(f"Normalized World Position Range: min={global_min}, max={global_max}")