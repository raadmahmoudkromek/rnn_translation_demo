import math
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import CrossEntropyLoss, Transformer
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.device import DEVICE


# A simple wrapper for an embedding layer from PyTorch.
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        # Forward pass through the embedding layer
        # mMultiplication by math.sqrt(self.emb_size) is a scaling factor used to stabilize the training process.
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Define a module for positional encoding. This module is responsible for creating and injecting positional information into token embeddings.
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, max_len: int = 5000, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Calculate positional embeddings
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, emb_size))

        # Calculate sinusoidal functions of different frequencies and embed them into the position tensor.
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        # Set dropout and register positional embeddings as buffer
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        # Apply positional encoding to token embeddings
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


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
        # Initialize Transformer layers, linear generator, token embeddings, and positional encoding
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
        # Forward pass through the Transformer model
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        # Generate output using linear layer
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        # Encoder pass through the Transformer
        return self.transformer.encoder(self.positional_encoding(
            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        # Decoder pass through the Transformer
        return self.transformer.decoder(self.positional_encoding(
            self.tgt_tok_emb(tgt)), memory, tgt_mask)


def training_epoch(model: Sequence2SequenceTransformer,
                   train_dataloader: DataLoader,
                   create_mask: Callable,
                   optimiser: Optimizer,
                   loss_func: CrossEntropyLoss) -> list[float]:
    # Set the model to training mode
    model.train()
    # Record losses during training
    loss_record = []

    for source, target in tqdm(train_dataloader):
        source = source.to(DEVICE)
        target = target.type(torch.LongTensor).to(DEVICE)

        # Prepare target input and output for training
        target_input = target[:-1, :]
        target_output = target[1:, :]

        source_mask, target_mask, source_padding_mask, target_padding_mask = create_mask(src=source,
                                                                                         tgt=target_input)

        # Forward pass through the model
        predictions = model(src=source, tgt=target_input,
                            src_mask=source_mask, tgt_mask=target_mask,
                            src_padding_mask=source_padding_mask, tgt_padding_mask=target_padding_mask,
                            memory_key_padding_mask=source_padding_mask)

        # Zero the gradients, compute loss, backward pass, and optimizer step
        optimiser.zero_grad()
        loss = loss_func(predictions.reshape(-1, predictions.shape[-1]), target_output.reshape(-1))
        loss.backward()
        optimiser.step()

        # Record the loss for this iteration
        loss_record.append(loss.item())

    # Return the list of losses
    return loss_record