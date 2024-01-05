import torch.nn as nn
from torch import Tensor
from torch.nn import CrossEntropyLoss, Transformer
from torch.optim import Optimizer
from torch.utils.data import DataLoader

class PositionalEncoding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError
    def forward(self):
        raise NotImplementedError

class TokenEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError
    def forward(self):
        raise NotImplementedError

class Sequence2SequenceTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 d_model: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 256,
                 dropout: float = 0.2):
        super(Sequence2SequenceTransformer, self).__init__()
        self.transformer = Transformer(d_model=d_model,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
            self.tgt_tok_emb(tgt)), memory, tgt_mask)


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
