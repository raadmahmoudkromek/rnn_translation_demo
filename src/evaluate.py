from typing import Callable

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from src.device import DEVICE
from src.model import Sequence2SequenceTransformer


def evaluate_model(model: Sequence2SequenceTransformer, val_dataloader: DataLoader, loss_fn: CrossEntropyLoss,
                   create_mask: Callable) -> float:
    model.eval()
    losses = 0

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.type(torch.LongTensor).to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))
