import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0, reduction: str = 'mean', device: str = DEVICE):
        super(FocalLoss, self).__init__()        
        self.gamma = gamma
        self.reduction = reduction
        self.to(device)
        if(alpha is None):
            self.register_buffer('alpha', None)
        else:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float, device=device))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss for multi-class classification."""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probabilities of the true class
        
        ce_loss_clamped = ce_loss.clamp(min=1e-8, max=100.0)
        
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss_clamped
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
