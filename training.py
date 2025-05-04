from typing import Optional, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from dataset import SingleSceneDataset
from duel_u_net import DuelUNet
from u_net import UNet
from mc_denoise_net import MCDenoiseNet, init_weights

from loss_functions import *
from exr_file_helper import *


DATASET_PATH = Path('../dataset')
INPUTS_FOLDER = 'inputs'
MODEL_PATH = Path('./models')
LOGS_PATH = Path('./training_logs')

SCENES = {
    'classroom': 'classroom',
    'living-room': 'living-room',
    'san-miguel': 'san-miguel',
    'sponza': 'sponza'
}

NUM_EPOCHS = 2500
VAL_RATIO = 0.2
BATCH_SIZE = 1
ACCUMULATION_STEPS = 8
ORIGIN_LEARNING_RATE = 5e-4
MIN_LEARNING_RATE = 5e-6

# feature_names = [SHADING_NORMAL, ALBEDO]
feature_names = [ALBEDO, SHADING_NORMAL, NORMALIZED_WORLD_POSITION]

# tensorboard --logdir=./training_logs

def load_all_scene_data():
    all_data = []
    for scene in SCENES.values():
        path = DATASET_PATH / scene / INPUTS_FOLDER
        data = SingleSceneDataset(str(path), feature_names)
        all_data.append(data)

    dataset = ConcatDataset(all_data)

    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size

    return random_split(dataset, [train_size, val_size])


def save_checkpoint(epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, best_val_loss: float, save_path: Path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss
    }, save_path)


def load_checkpoint(load_path: Path, device: torch.device, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer]) -> Tuple[int, float]:
    checkpoint = torch.load(load_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch'] + 1

    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    return start_epoch, best_val_loss


def train(base_model_path: Path=None, device: str='cuda', reset_lr: bool=False, scheduler_type: str='cos'):
    device = torch.device(device)

    feature_num = len(feature_names)
    model = MCDenoiseNet(3, feature_num * 3).to(device)
    model = init_weights(model)

    writer = SummaryWriter(log_dir=str(LOGS_PATH))

    optimizer = optim.Adam(model.parameters(), lr=ORIGIN_LEARNING_RATE)
    start_epoch = 0
    best_val_loss = float('inf')

    if base_model_path is not None:
        start_epoch, best_val_loss = load_checkpoint(base_model_path, device, model, optimizer)

    if reset_lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] = ORIGIN_LEARNING_RATE

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Starts with lr: {current_lr}")

    criterion = RelMSELoss()

    if scheduler_type == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50,
            eta_min=MIN_LEARNING_RATE
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.8,
            patience=3,
            min_lr=MIN_LEARNING_RATE
        )

    train_dataset, val_dataset = load_all_scene_data()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print('Start training.\n')
    for epoch in range(start_epoch, NUM_EPOCHS):
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate (log10)', torch.log10(torch.tensor(current_lr)), epoch)

        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for i, (image, features, target) in enumerate(train_loader):
            image = image.to(device)
            features = features.to(device)
            target = target.to(device)

            _, output = model(image, features)
            loss = criterion(output, target)
            loss /= ACCUMULATION_STEPS  # 梯度累积模拟大的batch size时，需要对每个mini batch的loss归一化
            loss.backward()
            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()

        train_loss = train_loss * ACCUMULATION_STEPS / len(train_loader)
        writer.add_scalar('Loss/Train', train_loss, epoch)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for image, features, target in val_loader:
                image = image.to(device)
                features = features.to(device)
                target = target.to(device)

                _, outputs = model(image, features)
                loss = criterion(outputs, target)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        writer.add_scalar('Loss/val', val_loss, epoch)

        # 保存在验证集上表现最好的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 保存模型
            save_checkpoint(epoch, model, optimizer, best_val_loss, MODEL_PATH / f'best_model.pth')
            print(f"Saved best model at epoch {epoch} with val_loss {val_loss:.4f}")

        # 调整学习率
        if scheduler_type == 'cos':
            scheduler.step()
        else:
            scheduler.step(val_loss)

        # 保存模型
        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if (epoch + 1) % 10 == 0:
            save_checkpoint(epoch, model, optimizer, best_val_loss, MODEL_PATH / f'Epoch{epoch}.pth')

    writer.close()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print('cuda is not available!')

    model_path = Path('models/Epoch1999.pth')
    train(
        base_model_path=model_path,
        device='cuda',
        reset_lr=False,
        scheduler_type='cos'
    )