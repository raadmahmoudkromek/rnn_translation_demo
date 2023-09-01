import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class Sequence2SequenceTransformer(nn.Module):
    raise NotImplementedError


def training_epoch(model: Sequence2SequenceTransformer,
                   train_dataloader: DataLoader,
                   optimiser: Optimizer,
                   loss_func: CrossEntropyLoss) -> list[float]:
    model.train()
    loss_record = []

    for source, target in train_dataloader:
        target_input = target[:-1, :]
        target_output = target[1:, :]

        predictions = model(src=source, trg=target_input)

        optimiser.zero_grad()

        loss = loss_func(predictions, target_output)
        loss.backward()

        optimiser.step()
        loss_record.append(loss.item())

    return loss_record
