import numpy as np
import torch
from torch.utils.data import Dataset

from exr_file_helper import *


class SingleSceneDataset(Dataset):
    def __init__(self, data_path: str, feature_names: list[str], transform=None):
        self.data_path = data_path
        self.feature_names = feature_names
        self.transform = transform
        self.cached_data = []

        self.cached_data = load_all_normalized_samples(data_path)

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx):
        sample = self.cached_data[idx]

        input = sample[INPUT]
        target = sample[TARGET]

        features = []
        for feature_name in self.feature_names:
            feature = sample[feature_name]
            features.append(feature)

        features = torch.cat(features, dim=0)

        return input, features, target


if __name__ == '__main__':
    path = "../dataset/classroom/inputs"
    #
    feature_names = [ALBEDO, SHADING_NORMAL, NORMALIZED_WORLD_POSITION]
    dataset = SingleSceneDataset(path, feature_names)

    _, features, _ = dataset[0]
    print(features.shape)