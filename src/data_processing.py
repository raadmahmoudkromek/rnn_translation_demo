import io
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from os import PathLike
from typing import Callable, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab, build_vocab_from_iterator


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
    vocabulary = vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'], special_first=True)
    vocabulary.set_default_index = 0
    return vocabulary


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
        self.vocab_b.set_default_index(0)

        self.pad_id = self.vocab_a['<pad>']
        self.bos_id = self.vocab_a['<bos>']
        self.eos_id = self.vocab_a['<eos>']

    @staticmethod
    def data_process(target_file_language_a: str | PathLike, target_file_language_b: str | PathLike):
        raw_a_iter = iter(io.open(target_file_language_a, encoding="utf8"))
        raw_b_iter = iter(io.open(target_file_language_b, encoding="utf8"))
        for (raw_a, raw_b) in zip(raw_a_iter, raw_b_iter):
            yield raw_a, raw_b

    def lang_a_text_transform(self, item):
        item = self.tokenizer_a(item)
        item = self.vocab_a(item)
        item = torch.cat(
            [torch.tensor([self.bos_id]),
             torch.tensor(item),
             torch.tensor([self.eos_id])]
        )
        return item

    def lang_b_text_transform(self, item):
        item = self.tokenizer_b(item)
        item = self.vocab_b(item)
        item = torch.cat(
            [torch.tensor([self.bos_id]),
             torch.tensor(item),
             torch.tensor([self.eos_id])]
        )
        return item

    def collation(self, batch, ):
        a_batch, b_batch = [], []
        for (a_item, b_item) in batch:
            a_batch.append(self.lang_a_text_transform(a_item.rstrip("\n")))
            b_batch.append(self.lang_b_text_transform(b_item.rstrip("\n")))
        a_batch = pad_sequence(a_batch, padding_value=self.pad_id)
        b_batch = pad_sequence(b_batch, padding_value=self.pad_id)
        return a_batch, b_batch
