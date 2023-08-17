import yaml
from torch.utils.data import DataLoader

from src.data_processing import DataProcessor

# Open our general config file.
with open('config.yml', 'r') as configfile:
    config = yaml.load(configfile, Loader=yaml.SafeLoader)

# Instantiate a DataProcessor object for the languages we have specified.
data_processor = DataProcessor(
    language_a=config['language_a'],
    language_b=config['language_b'],
    train_file_language_a=config['language_a_train_file'],
    train_file_language_b=config['language_b_train_file']
)

#Prepare a training dataset using our DataProcessor instance
train_iter = data_processor.data_process(
    target_file_language_a=config['language_a_train_file'],
    target_file_language_b=config['language_b_train_file']
)


# Create a training data iterator to be used for the training loop
train_dataloader = DataLoader(
    list(train_iter),
    batch_size=config['training']['batch_size'],
    shuffle=True,
    collate_fn=data_processor.collation
)