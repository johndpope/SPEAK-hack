import torch
import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from CIPS2 import CIPSskip
# Initialize the model
generator = CIPSskip(size=256, hidden_size=512, style_dim=512, n_mlp=8)  # Adjust parameters as needed


checkpoint = torch.load('/media/oem/12TB/SPEAK-hack/reference/CIPS/ffhq1024_g_ema.pt')

# Load the state dict into the model
generator.load_state_dict(checkpoint)

# Set the model to evaluation mode
generator.eval()

print("Checkpoint loaded successfully!")
