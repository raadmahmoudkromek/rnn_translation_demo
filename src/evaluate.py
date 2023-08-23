from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from src.model import Sequence2SequenceTransformer


def evaluate_model(model: Sequence2SequenceTransformer,
                   val_dataloader: DataLoader,
                   loss_func: CrossEntropyLoss) -> float:
    raise NotImplementedError
