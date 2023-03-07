import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(MemoryLoss, self).__init__()
        self.eps = eps
    def forward(self, memory1, memory2):
        memory1 = F.normalize(memory1)
        memory2 = F.normalize(memory2)
        dot_production = torch.mm(memory1, memory2.T)
        dot_size = dot_production.size(0)
        loss = dot_production.norm()

        return loss


