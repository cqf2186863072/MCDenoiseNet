import torch
from torch.profiler import profile
import time
import numpy as np
from pathlib import Path

from duel_u_net import DuelUNet
from u_net import UNet
from mc_denoise_net import MCDenoiseNet
from training import load_checkpoint
from exr_file_helper import *
from loss_functions import *
from png2exr import *

def save_detail(image, filename, crop_coords=(0, 0, 64, 64)):
    """裁剪并保存图像的一部分"""
    image = image[:, crop_coords[1]:crop_coords[3], crop_coords[0]:crop_coords[2]]
    save_exr(image, filename)

if __name__ == '__main__':
    scene_name = 'classroom'
    prefix = 'Classroom'
    idx = '001'
    detail_crop_coords = (570, 320, 670, 420)

    samples_path = "../dataset/" + scene_name + "/inputs"
    outputs_path = "outputs/" + scene_name + '/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sample = load_single_sample(samples_path, int(idx) - 1)
    color = sample[COLOR]
    albedo = sample[ALBEDO]
    normal = sample[SHADING_NORMAL]
    world_position = sample[WORLD_POSITION]

    target = sample[REFERENCE]
    save_exr(target, outputs_path + "target.exr")
    save_detail(target, outputs_path + 'details/' + "target.exr", detail_crop_coords)
    target = target.to(device).unsqueeze(0)

    pos_mean = world_position.mean()
    pos_std = world_position.std()
    norm_pos = (world_position - pos_mean) / (pos_std + 1e-6)
    features = torch.cat([albedo, normal, norm_pos], dim=0)
    features = features.to(device).unsqueeze(0)

    input = color * albedo
    save_exr(input, outputs_path + "input.exr")
    save_detail(input, outputs_path + 'details/' + "input.exr", detail_crop_coords)

    input = input.to(device).unsqueeze(0)

    #duel encoder UNet

    model_path = Path('models/best_models/duel_unet_best_model.pth')
    model = MCDenoiseNet(3, 9).to(device)
    load_checkpoint(model_path, device, model, None)
    model.eval()

    with torch.no_grad():
        _, _ = model(input, features)

        with profile(with_stack=True) as prof:
            feature_map, output = model(input, features)

    # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=50))

    loss_fuc = RelMSELoss()
    loss = loss_fuc(output, target)
    print(f"Ours RelMSE: {loss.item():.6f}")
    loss_fuc = SSIM()
    loss = loss_fuc(output, target)
    print(f"Ours SSIM: {1-loss.item():.6f}")

    feature_map = feature_map.squeeze(0)
    output = output.squeeze(0)

    save_exr(feature_map, outputs_path + "de_feature_map.exr")
    save_exr(output, outputs_path + "de_output.exr")
    save_detail(output, outputs_path + 'details/' + "de_output.exr", detail_crop_coords)

    png_file_name = "../dataset/" + scene_name + '/' + 'Reference' + '/' + prefix + '-' + 'Reference' + '-' + idx + '.png'
    exr_file_name = outputs_path + 'Reference' + '.exr'
    png_to_exr(png_file_name, exr_file_name, True)
    reference = load_exr(exr_file_name).to(device).unsqueeze(0)

    other_methods = ['BMFR', 'Guided filter', 'NFOR', 'ONND', 'SVGF']
    for method_name in other_methods:
        png_file_name = "../dataset/" + scene_name + '/' + method_name + '/' + prefix + '-' + method_name + '-' + idx + '.png'
        exr_file_name = outputs_path + method_name + '.exr'
        png_to_exr(png_file_name, exr_file_name, True)
        img = load_exr(exr_file_name)
        save_detail(img, outputs_path + 'details/' + method_name + '.exr', detail_crop_coords)
        img = img.to(device).unsqueeze(0)

        loss_fuc = RelMSELoss()
        loss = loss_fuc(img, reference)
        print(f"{method_name} RelMSE: {loss.item():.6f}")
        loss_fuc = SSIM()
        loss = loss_fuc(img, reference)
        print(f"{method_name} SSIM: {1 - loss.item():.6f}")


