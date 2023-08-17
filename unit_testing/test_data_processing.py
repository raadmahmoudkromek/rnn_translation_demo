import pytest
from unittest.mock import Mock, mock_open, patch, MagicMock
from src.data_processing import build_vocab, DataProcessor

@pytest.fixture
def sample_data(tmpdir):
    file_a = tmpdir.join("train_a.txt")
    file_b = tmpdir.join("train_b.txt")
    file_a.write("This is a sample sentence.\nAnother sentence here.\n")
    file_b.write("Translated sentence 1.\nAnother translated sentence here.\n")
    return str(file_a), str(file_b)

@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()
    tokenizer.side_effect = lambda x: x.split()
    return tokenizer

def test_build_vocab(mock_tokenizer, sample_data):
    vocab_a = build_vocab(sample_data[0], mock_tokenizer)
    assert len(vocab_a) == 12  # Number of unique tokens in the sample data
    vocab_b = build_vocab(sample_data[1], mock_tokenizer)
    assert len(vocab_b) == 10  # Number of unique tokens in the sample data

def test_data_processor_init(sample_data):
    dp = DataProcessor("en_core_web_sm", "es_core_news_sm", *sample_data)
    assert dp.vocab_a is not None
    assert dp.vocab_b is not None
    assert dp.pad_id == dp.vocab_a['<pad>']
    assert dp.bos_id == dp.vocab_a['<bos>']
    assert dp.eos_id == dp.vocab_a['<eos>']

def test_data_process(sample_data, tmpdir):
    dp = DataProcessor("en_core_web_sm", "es_core_news_sm", *sample_data)
    target_file_a = tmpdir.join("target_a.txt")
    target_file_b = tmpdir.join("target_b.txt")
    target_file_a.write("Target A 1\nTarget A 2\n")
    target_file_b.write("Target B 1\nTarget B 2\n")

    result = list(dp.data_process(str(target_file_a), str(target_file_b)))
    assert len(result) == 2
    assert result[0] == ("Target A 1\n", "Target B 1\n")
    assert result[1] == ("Target A 2\n", "Target B 2\n")

def test_generate_batch(sample_data):
    dp = DataProcessor("en_core_web_sm", "es_core_news_sm", *sample_data)
    mock_batch = [("This is a sentence.", "Translated sentence.")]
    a_batch, b_batch = dp.collation(mock_batch)
    assert a_batch.shape[1] == b_batch.shape[1]  # Both should have the same sequence length

def test_generate_batch_padding(sample_data):
    dp = DataProcessor("en_core_web_sm", "es_core_news_sm", *sample_data)
    mock_batch = [("This is a sentence.", "Translated sentence.")]
    a_batch, b_batch = dp.collation(mock_batch)
    assert a_batch[0][0] == dp.bos_id
    assert a_batch[-1][0] == dp.eos_id
    assert b_batch[0][0] == dp.bos_id
    assert b_batch[-1][0] == dp.eos_id
#
# def test_generate_batch_tokenization(sample_data):
#     dp = DataProcessor("en_core_web_sm", "es_core_news_sm", *sample_data)
#     mock_batch = [("This is a sentence.", "Translated sentence.")]
#     a_batch, b_batch = dp.collation(mock_batch)
#     assert a_batch[1][1] == dp.vocab_a["is"]  # Token "is" in the vocab
#     assert b_batch[1][1] == dp.vocab_b["Translated"]  # Token "Translated" in the vocab