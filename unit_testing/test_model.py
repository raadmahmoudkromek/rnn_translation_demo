import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from src.data_processing import DataProcessor, pairwise_sentence_iterator
from src.device import DEVICE
from src.model import Sequence2SequenceTransformer, training_epoch


@pytest.fixture
def prepare_data():
    with open('test_config.yml', 'r') as configfile:
        config = yaml.load(configfile, Loader=yaml.SafeLoader)
    # Instantiate a DataProcessor object for the languages we have specified.
    data_processor = DataProcessor(
        language_a=config['language_a'],
        language_b=config['language_b'],
        train_file_language_a=config['language_a_train_file'],
        train_file_language_b=config['language_b_train_file']
    )

    # Prepare the training dataset iterator
    train_iter = pairwise_sentence_iterator(
        target_file_language_a=config['language_a_train_file'],
        target_file_language_b=config['language_b_train_file']
    )

    # Create a training data DataLoader to be used for the training loop
    train_dataloader = DataLoader(
        list(train_iter),
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=data_processor.collation
    )

    return train_dataloader, data_processor


@pytest.fixture
def prepare_model(prepare_data):
    with open('test_config.yml', 'r') as configfile:
        config = yaml.load(configfile, Loader=yaml.SafeLoader)

    transformer = Sequence2SequenceTransformer(num_encoder_layers=config['model']['num_encoder_layers'],
                                               num_decoder_layers=config['model']['num_decoder_layers'],
                                               d_model=config['model']['embedding_size'],
                                               nhead=config['model']['num_heads'],
                                               src_vocab_size=len(prepare_data[1].vocab_a),
                                               tgt_vocab_size=len(prepare_data[1].vocab_b),
                                               dim_feedforward=config['model']['feed_forward_hidden_dimensions'],
                                               dropout=config['model']['dropout_rate'])
    transformer = transformer.to(DEVICE)

    optimiser = torch.optim.Adam(transformer.parameters(), lr=0.0001,
                                 betas=(0.9, 0.98), eps=1e-9)
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=prepare_data[1].pad_id)

    return transformer, optimiser, loss_func


def test_training_epoch(prepare_data, prepare_model):
    train_dataloader, data_processor = prepare_data
    transformer, optimiser, loss_func = prepare_model
    train_loss_epoch_1 = training_epoch(transformer,
                                        train_dataloader,
                                        create_mask=data_processor.create_mask,
                                        optimiser=optimiser,
                                        loss_func=loss_func)
    train_loss_epoch_2 = training_epoch(transformer,
                                        train_dataloader,
                                        create_mask=data_processor.create_mask,
                                        optimiser=optimiser,
                                        loss_func=loss_func)
    assert np.average(train_loss_epoch_2) < np.average(train_loss_epoch_1)
