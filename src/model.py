import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class Sequence2SequenceTransformer(nn.Module):
    raise NotImplementedError


def training_iteration(model: Sequence2SequenceTransformer,
                       train_dataloader: DataLoader,
                       optimiser: Optimizer,
                       loss_func: CrossEntropyLoss) -> float:
    raise NotImplementedError
