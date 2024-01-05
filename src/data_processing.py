import io
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from os import PathLike
from typing import Callable, Generator

import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab

from src.device import DEVICE


def yield_tokens(data_iterator: Iterable, tokenizer: Callable) -> Generator[list[str]]:
    """
    Generator which takes the next element in a sentence iterable, tokenize it, and yield this tokenized value.
    Args:
        data_iterator: An iterator over strings.
        tokenizer: A tokenizer such as a spacy pipeline

    """
    for data_sample in data_iterator:
        yield tokenizer(data_sample)


def pairwise_sentence_iterator(target_file_language_a: str | PathLike, target_file_language_b: str | PathLike) -> \
        Generator[str, str]:
    """
    Generator which returns paired elements from two language files.
    Args:
        target_file_language_a: Corpus text file corresponding to language a
        target_file_language_b: Corpus text file corresponding to language b

    """
    raw_a_iter = iter(io.open(target_file_language_a, encoding="utf8"))
    raw_b_iter = iter(io.open(target_file_language_b, encoding="utf8"))
    for (raw_a, raw_b) in zip(raw_a_iter, raw_b_iter):
        yield raw_a, raw_b


@dataclass
class DataProcessor:
    """
    Defines a processor object that stores salient information about the language we translate from (language a) and
    the language we translate to (language b).
    """
    language_a: str  # The language of choice we wish to translate FROM, as a spacy pipeline e.g. "en_core_web_sm"
    language_b: str  # The language we wish to translate TO, as a spacy pipeline e.g. "es_core_news_sm"
    train_file_language_a: str  # A file consisting of a single sentence per line in language a
    train_file_language_b: str  # A file consisting of the same sentences as "train_file_language_a", but in language b

    def __post_init__(self):
        partial_build_vocab_from_iterator = partial(build_vocab_from_iterator, min_freq=1,
                                                    specials=['<unk>', '<pad>', '<bos>', '<eos>'], special_first=True)

        self.tokenizer_a = get_tokenizer('spacy', language=self.language_a)
        self.tokenizer_b = get_tokenizer('spacy', language=self.language_b)

        with iter(io.open(self.train_file_language_a, 'r')) as train_iter_a:
            self.vocab_a = partial_build_vocab_from_iterator(yield_tokens(train_iter_a, tokenizer=self.tokenizer_a))
        self.vocab_a.set_default_index(0)
        with iter(io.open(self.train_file_language_b, 'r')) as train_iter_b:
            self.vocab_b = partial_build_vocab_from_iterator(yield_tokens(train_iter_b, tokenizer=self.tokenizer_b))
        self.vocab_b.set_default_index(0)

        self.pad_id = self.vocab_a['<pad>']
        self.bos_id = self.vocab_a['<bos>']
        self.eos_id = self.vocab_a['<eos>']

    def lang_text_transform(self, item, tokenizer: Callable, vocab: Vocab) -> torch.Tensor:
        """
        Function to take a language element, tokenize it, map the tokens to indices, and prepare a tensor.
        """
        item = tokenizer(item)
        item = vocab(item)
        item = torch.cat(
            [torch.tensor([self.bos_id]),
             torch.tensor(item),
             torch.tensor([self.eos_id])]
        )
        return item

    def collation(self, batch: list[tuple[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Function to prepare and pad tensors from a batch of paired string data.
        Args:
            batch: A batch of paired string data.
        """
        a_batch, b_batch = [], []
        for (a_item, b_item) in batch:
            a_batch.append(
                self.lang_text_transform(a_item.rstrip("\n"), tokenizer=self.tokenizer_a, vocab=self.vocab_a))
            b_batch.append(
                self.lang_text_transform(b_item.rstrip("\n"), tokenizer=self.tokenizer_a, vocab=self.vocab_a))
        a_batch = pad_sequence(a_batch, padding_value=self.pad_id)
        b_batch = pad_sequence(b_batch, padding_value=self.pad_id)
        return a_batch, b_batch

    def create_mask(self, src: torch.Tensor, tgt: torch.Tensor):
        """
        Function to create a set of masks for the masking of padding tokens and the implementation of the transformer
        self-attention mechanism, particularly to prevent the model from looking at tokens at future time-steps when
        making predictions.
        Args:
            src: A pytorch tensor for the source language.
            tgt: A pytorch tensor for the target language.

        Returns:
            src_mask, used during the encoder self-attention step
            tgt_mask, used during the decoder self-attention step
            src_padding_mask, used to indicate padding positions in the src sequence
            tgt_padding_mask, used to indicate padding positions in the tgt sequence
        """

        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

        src_padding_mask = (src == self.pad_id).transpose(0, 1)
        tgt_padding_mask = (tgt == self.pad_id).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def generate_square_subsequent_mask(size: int):
    """
    Function which takes a sequence length (size), and creates a square mask for the self-attention mechanism decoder mechanism.
    It first creates a square matrix of ones, which it then converts to an upper triangular matrix (torch.triu).
    It then sets all 0-valued elements to -inf and all 1-valued elements to zero, effectively masking future positions.
    Args:
        size:

    Returns:

    """
    mask = (torch.triu(torch.ones((size, size), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
