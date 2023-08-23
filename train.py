import yaml
from torch.utils.data import DataLoader

from src.data_processing import DataProcessor
from src.evaluate import evaluate_model
from src.model import Sequence2SequenceTransformer, training_iteration

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

# Prepare a training dataset using our DataProcessor instance
train_iter = data_processor.data_process(
    target_file_language_a=config['language_a_train_file'],
    target_file_language_b=config['language_b_train_file']
)

# Prepare the testing dataset using our DataProcessor instance
test_iter = data_processor.data_process(
    target_file_language_a=config['language_a_test_file'],
    target_file_language_b=config['language_b_test_file']
)

# Create a training data DataLoader to be used for the training loop
train_dataloader = DataLoader(
    list(train_iter),
    batch_size=config['training']['batch_size'],
    shuffle=True,
    collate_fn=data_processor.collation
)

# Create a test data DataLoader to be used for evaluation
test_dataloader = DataLoader(
    list(test_iter),
    batch_size=config['training']['batch_size'],
    shuffle=True,
    collate_fn=data_processor.collation
)

transformer = Sequence2SequenceTransformer()

for epoch in range(1, config['training']['num_epochs'] + 1):
    training_iteration(transformer, train_dataloader, optimiser=optimiser, loss_func=loss_func)
    evaluate_model(transformer, test_dataloader, loss_func=loss_func)
