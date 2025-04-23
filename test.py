import exr_file_helper
from exr_file_helper import *

from PIL import Image
import torchvision.transforms as transforms
import torch
from loss_functions import *


OTHER_METHORDS = {
    'BMFR': 'BMFR',
    'Guided filter': 'Guided filter',
    'NFOR': 'NFOR',
    'ONND': 'ONND',
    'SVGF': 'SVGF'
}

SCENES = {
    'classroom': 'Classroom',
    'living-room': 'Living-room',
    'san-miguel': 'San-miguel',
    'sponza': 'Sponza'
}


def test_aces():
    dataset_path = "../dataset/classroom/"

    device = torch.device('cuda')

    sample = load_single_sample(dataset_path + 'inputs', 0)

    color = sample[COLOR]
    albedo = sample[ALBEDO]
    reference = sample[REFERENCE]

    combined = color * albedo

    reference = reference
    acesed = apply_aces_with_gamma(reference)

    print("Reference max:", torch.max(reference).item())  # 输出reference的最大像素值
    print("ACESed max:", torch.max(acesed).item())  # 输出acesed的最大像素值

    save_exr(acesed, './aces.exr')
    save_exr(reference, './reference.exr')

def load_image_as_tensor(image_path):
    """
    读取 PNG 图像并转换为 PyTorch tensor（归一化到 [0, 1]）
    """
    pass


def compute_diff():
    dataset_path = "../dataset/classroom/"

    device = torch.device('cuda')
    loss_fuc = MSELossWithSSIM()

    loss = 0
    for i in range(60):
        sample = load_single_sample(dataset_path + 'inputs', i)

        color = sample[COLOR]
        albedo = sample[ALBEDO]
        reference = sample[REFERENCE]

        combined = color * albedo

        combined = combined.to(device).unsqueeze(0)
        reference = reference.to(device).unsqueeze(0)
        loss += loss_fuc(combined, reference).item()

    loss /= 60
    print(f"loss with noise:", loss)


if __name__ == '__main__':
    compute_diff()
    # test_aces()