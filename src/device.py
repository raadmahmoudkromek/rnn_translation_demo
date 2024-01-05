#  Copyright (c) 2023. Kromek Group Ltd.

import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
