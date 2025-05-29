import torch
import torch.nn as nn
import torchvision.models
from pytorch_msssim import ssim
import lpips

class MSELossWithSSIM(nn.Module):
    def __init__(self, alpha=0.9): # alpha越大越偏向MSE
        super(MSELossWithSSIM, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        normalized_pred = apply_aces_with_gamma(pred)
        normalized_target = apply_aces_with_gamma(target)

        mse_loss = self.mse(pred, target)
        ssim_loss = 1 - ssim(normalized_pred, normalized_target, data_range=1.0, size_average=True)

        # print(mse_loss)
        # print(ssim_loss)
        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss


class SSIM(nn.Module):
    def __init__(self, alpha=0.9): # alpha越大越偏向MSE
        super(SSIM, self).__init__()

    def forward(self, pred, target):
        normalized_pred = apply_aces_with_gamma(pred)
        normalized_target = apply_aces_with_gamma(target)

        ssim_loss = 1 - ssim(normalized_pred, normalized_target, data_range=1.0, size_average=True)

        return ssim_loss

class RelMSELoss(nn.Module):
    def __init__(self, epsilon=1e-3, reduction='mean'):
        super(RelMSELoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, pred, target):
        diff_sq = (pred - target) ** 2
        denom = target ** 2 + self.epsilon
        rel_mse = diff_sq / denom

        if self.reduction == 'mean':
            return rel_mse.mean()
        elif self.reduction == 'sum':
            return rel_mse.sum()
        else:
            print('Param reduction should be \'mean\' or \'sum\'.')
            return rel_mse


class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        self.loss_fuc = lpips.LPIPS(net='vgg').to(device)

    def forward(self, pred, target):
        loss = self.loss_fuc(pred, target).mean()
        return loss


def aces(x: torch.Tensor) -> torch.Tensor:
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14

    numerator = x * (a * x + b)
    denominator = x * (c * x + d) + e

    return torch.clamp(numerator / denominator, 0.0, 1.0)


def apply_aces_with_gamma(x: torch.Tensor, gamma=2.2) -> torch.Tensor:
    linear_x = x ** gamma
    aces_x = aces(linear_x)
    return aces_x ** (1/gamma)