# ======================================================================================
# ===================================== libraries ==========================================
# ======================================================================================

import os
import pywt
import torch
import numpy as np
import pandas as pd
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data_utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from sklearn.preprocessing import LabelEncoder
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support, \
    roc_auc_score

import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
from scipy.signal import butter, filtfilt, stft
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")

# Set environment variable to reduce memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
