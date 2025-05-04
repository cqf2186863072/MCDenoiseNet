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

if __name__ == '__main__':
    samples_path = "../dataset/classroom/inputs"
    # model_path = Path('models/best_model.pth')
    model_path = Path('models/Epoch2349.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model = MCDenoiseNet(3, 9).to(device)
    loss_fuc = RelMSELoss()

    load_checkpoint(model_path, device, model, None)

    model.eval()

    sample = load_single_sample(samples_path, 0)
    color = sample[COLOR]
    albedo = sample[ALBEDO]
    normal = sample[SHADING_NORMAL]
    world_position = sample[WORLD_POSITION]

    input = color * albedo

    pos_mean = world_position.mean()
    pos_std = world_position.std()
    norm_pos = (world_position - pos_mean) / (pos_std + 1e-6)

    features = torch.cat([albedo, normal, norm_pos], dim=0)
    target = sample[REFERENCE]

    input = input.to(device).unsqueeze(0)
    features = features.to(device).unsqueeze(0)
    target = target.to(device).unsqueeze(0)

    with torch.no_grad():
        _, _ = model(input, features)

        with profile(with_stack=True) as prof:
            feature_map, output = model(input, features)

    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=50))

    loss = loss_fuc(output, target)

    feature_map = feature_map.squeeze(0)
    output = output.squeeze(0)

    print(f"Loss: {loss.item():.6f}")



    save_exr(feature_map, "./feature_map.exr")
    save_exr(output, "./model_output.exr")