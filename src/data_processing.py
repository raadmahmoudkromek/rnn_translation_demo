import io
from collections import Counter
from dataclasses import dataclass
from os import PathLike

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab


def build_vocab(filepath, tokenizer):
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            counter.update(tokenizer(string_))
    return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


@dataclass
class DataProcessor:
    language_a: str  # The language of choice we wish to translate FROM, e.g. "en"
    language_b: str  # The language we wish to translate TO, e.g. "es"
    train_file_language_a: str  # A file consisting of a single sentence per line in language a
    train_file_language_b: str  # A file consisting of the same sentences as "train_file_language_a", but in language b

    def __post_init__(self):
        self.tokenizer_a = get_tokenizer('spacy', language=self.language_a)
        self.tokenizer_b = get_tokenizer('spacy', language=self.language_b)
        self.vocab_a = build_vocab(self.train_file_language_a, self.tokenizer_a)
        self.vocab_b = build_vocab(self.train_file_language_b, self.tokenizer_b)

        self.pad_id = self.vocab_a['<pad>']

    def data_process(self, target_file_language_a: str | PathLike, target_file_language_b: str | PathLike):
        raw_a_iter = iter(io.open(target_file_language_a, encoding="utf8"))
        raw_b_iter = iter(io.open(target_file_language_b, encoding="utf8"))
        data = []
        for (raw_a, raw_b) in zip(raw_a_iter, raw_b_iter):
            tensor_a_ = torch.tensor([self.vocab_a[token] for token in self.tokenizer_a(raw_a)], dtype=torch.long)
            tensor_b_ = torch.tensor([self.vocab_b[token] for token in self.tokenizer_b(raw_b)], dtype=torch.long)
            data.append((tensor_a_, tensor_b_))
        return data

def produce_batch(tensor_batch):
    en_batch, es_batch = [], []
    for (en_item, es_item) in tensor_data:
        en_batch.append(torch.)

