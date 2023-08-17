import io
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from typing import Callable, List
from os import PathLike

import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab, build_vocab_from_iterator


def build_vocab(filepath, tokenizer):
    """
    Function for counting instances of tokenized words in a language dataset
    Args:
        filepath: Path to file containing language examples
        tokenizer: A tokenization pipeline such as a spacy pipeline.

    Returns:
        A torchtext.vocab.Vocab object which numericalises a language case.
    """
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            counter.update(tokenizer(string_))
    vocab = Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'], specials_first=True)
    vocab.set_default_index = 0
    return vocab


def yield_tokens(data_iterator: Iterable, tokenizer: Callable) -> List[str]:
    for data_sample in data_iterator:
        yield tokenizer(data_sample)


@dataclass
class DataProcessor:
    """
    Defines a processor object that stores salient information about the language we translate from (language a) and
    the language we translate to (language b).
    """
    language_a: str  # The language of choice we wish to translate FROM, as a spacy pipeline e.g. "en_core_web_sm"
    language_b: str  # The language we wish to translate TO, , as a spacy pipeline e.g. "es_core_news_sm"
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
        self.vocab_b.set_default_index

        self.pad_id = self.vocab_a['<pad>']
        self.bos_id = self.vocab_a['<bos>']
        self.eos_id = self.vocab_a['<eos>']

    def data_process(self, target_file_language_a: str | PathLike, target_file_language_b: str | PathLike):
        raw_a_iter = iter(io.open(target_file_language_a, encoding="utf8"))
        raw_b_iter = iter(io.open(target_file_language_b, encoding="utf8"))
        for (raw_a, raw_b) in zip(raw_a_iter, raw_b_iter):
            yield(raw_a, raw_b)

    def generate_batch(self, batch, ):
        a_batch, b_batch = [], []
        for (a_item, b_item) in batch:
            a_item = self.tokenizer_a(a_item.rstrip("\n"))
            b_item = self.tokenizer_b(b_item.rstrip("\n"))
            a_item = self.vocab_a(a_item)
            b_item = self.vocab_b(b_item))
            a_batch.append(torch.cat(
                [torch.tensor([self.bos_id]),
                 torch.tensor(a_item),
                 torch.tensor([self.eos_id])],
                dim=0
            ))
            b_batch.append(torch.cat(
                [torch.tensor([self.bos_id]),
                 torch.tensor(b_item),
                 torch.tensor([self.eos_id])],
                dim=0
            ))
        a_batch = pad_sequence(a_batch, padding_value=self.pad_id)
        b_batch = pad_sequence(b_batch, padding_value=self.pad_id)
        return a_batch, b_batch
