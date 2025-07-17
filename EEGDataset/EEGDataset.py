import os
import pywt
import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

from torch.utils.data import Dataset, DataLoader, TensorDataset

from Thesis.filters.bandpass_filter import bandpass_filter
from Thesis.spectrograms.spectrogram import create_stft_spectrogram


# ************************** Fusion Dataset **************************

class FusionDataset(Dataset):
    CLASSES = ['Seizure', 'LPD']
    CHANNELS = 19

    def __init__(self, ids, base_path, target_path, window_size=200, step_size=50, n_component=2, transform=None):
        self.id_list = ids
        self.base_path = base_path
        self.target_path = target_path
        self.window_size = window_size
        self.step_size = step_size
        self.transform = transform
        self.n_component = n_component

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.CLASSES)

        self.df = pd.read_csv(self.target_path)
        self.labels_df = self.df[self.df['target'].isin(self.CLASSES)]
        self.labels_dict = dict(zip(self.labels_df['eeg_id'].astype(str),
                                    self.label_encoder.transform(self.labels_df['target'])))

    def process_channel_data(self, data):
        """Process individual channel data with bandpass filter"""
        if isinstance(data, pd.Series):
            data = data.values
        return bandpass_filter(data, lowcut=0.5, highcut=30, fs=200, order=4)

    def sliding_window(self, signal):
        windows = []
        for i in range(0, len(signal) - self.window_size + 1, self.step_size):
            window = signal[i:i + self.window_size]
            windows.append(window)
        return np.array(windows)

    def __getitem__(self, index):
        file_id = self.id_list[index]
        file_path = os.path.join(self.base_path, f"{file_id}.parquet")

        try:
            # Load and preprocess data
            data = pd.read_parquet(file_path).drop('EKG', axis=1)
            center = len(data) // 2
            start, end = center - 5000, center + 5000

            # Ensure we have enough data
            if end - start < 10000:
                raise ValueError(f"Insufficient data length for file {file_id}")

            central_10_seconds = data.iloc[start:end, :]

            # Apply bandpass filter to each channel separately
            filtered_data = pd.DataFrame(
                {col: self.process_channel_data(central_10_seconds[col])
                 for col in central_10_seconds.columns},
                index=central_10_seconds.index
            )

            # Standardize the data
            scaler = StandardScaler()
            df_scaled = pd.DataFrame(
                scaler.fit_transform(filtered_data),
                columns=filtered_data.columns
            )

            # Initialize lists for processed data
            lstm_data = []
            cnn_data = []

            # Process each channel group
            for key, channels in self.CHANNELS.items():
                n_components = self.n_component if len(channels) > 3 else 1
                channel_data = df_scaled[channels]

                # Apply PCA
                pca = PCA(n_components=n_components)
                df_pca = pca.fit_transform(channel_data)

                for i in range(n_components):
                    # Process for LSTM
                    windowed_data = self.sliding_window(df_pca[:, i])
                    lstm_data.append(windowed_data)

                    # Process for CNN
                    stft_spec = create_stft_spectrogram(df_pca[:, i])
                    cnn_data.append(stft_spec)

            # Convert to tensors
            lstm_tensor = torch.tensor(np.stack(lstm_data, axis=0), dtype=torch.float32)
            cnn_tensor = torch.tensor(np.stack(cnn_data, axis=0), dtype=torch.float32)

            if self.transform:
                lstm_tensor = self.transform(lstm_tensor)
                cnn_tensor = self.transform(cnn_tensor)

            label = self.labels_dict[str(file_id)]
            return cnn_tensor, lstm_tensor, label

        except Exception as e:
            print(f"Error processing file {file_id}: {str(e)}")

    def __len__(self):
        return len(self.id_list)

    @classmethod
    def get_num_classes(cls):
        return len(cls.CLASSES)

    def get_class_weights(self):
        labels = np.array([self.labels_dict[str(id)] for id in self.id_list])
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        weights = total_samples / (len(self.CLASSES) * class_counts)
        return torch.FloatTensor(weights)


# ************************** CNN Dataset **************************

class CNNDataset(Dataset):
    CLASSES = ['Seizure', 'LPD']
    CHANNELS = 19

    def __init__(self, ids, base_path, target_path, transform=None):
        self.id_list = ids
        self.base_path = base_path
        self.target_path = target_path
        self.transform = transform

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.CLASSES)

        self.df = pd.read_csv(self.target_path)
        self.labels_df = self.df[self.df['target'].isin(self.CLASSES)]
        self.labels_dict = dict(zip(self.labels_df['eeg_id'].astype(str),
                                    self.label_encoder.transform(self.labels_df['target'])))

    def process_channel_data(self, data):
        """Process individual channel data with bandpass filter"""
        if isinstance(data, pd.Series):
            data = data.values
        return bandpass_filter(data, lowcut=0.5, highcut=30, fs=200, order=4)

    def __getitem__(self, index):
        file_id = self.id_list[index]
        file_path = os.path.join(self.base_path, f"{file_id}.parquet")

        try:
            # Load and preprocess data
            data = pd.read_parquet(file_path).drop('EKG', axis=1)
            center = len(data) // 2
            start, end = center - 5000, center + 5000

            # Ensure we have enough data
            if end - start < 10000:
                raise ValueError(f"Insufficient data length for file {file_id}")

            central_10_seconds = data.iloc[start:end, :]

            # Apply bandpass filter to each channel separately
            filtered_data = pd.DataFrame(
                {col: self.process_channel_data(central_10_seconds[col])
                 for col in central_10_seconds.columns},
                index=central_10_seconds.index
            )

            # Standardize the data
            scaler = StandardScaler()
            df_scaled = pd.DataFrame(
                scaler.fit_transform(filtered_data),
                columns=filtered_data.columns
            )

            cnn_data = []

            # Process each channel group
            for i in range(self.CHANNELS):
                channel_data = df_scaled.iloc[:, i]
                stft_spec = create_stft_spectrogram(channel_data)
                cnn_data.append(stft_spec)

            # Convert to tensors
            cnn_tensor = torch.tensor(np.stack(cnn_data, axis=0), dtype=torch.float32)

            if self.transform:
                cnn_tensor = self.transform(cnn_tensor)

            label = self.labels_dict[str(file_id)]
            return cnn_tensor, label

        except Exception as e:
            print(f"Error processing file {file_id}: {str(e)}")

    def __len__(self):
        return len(self.id_list)

    @classmethod
    def get_num_classes(cls):
        return len(cls.CLASSES)

    def get_class_weights(self):
        labels = np.array([self.labels_dict[str(id)] for id in self.id_list])
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        weights = total_samples / (len(self.CLASSES) * class_counts)
        return torch.FloatTensor(weights)




# ************************** Continuous Wavelet Transform (CWT) **************************
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
import warnings


# ************************** STFT Function (Updated) **************************

def create_stft_spectrogram_torch(signal: torch.Tensor, n_fft: int = 512, hop_length: int = 128,
                                  win_length: int = 512) -> torch.Tensor:
    """
    Convert a single EEG channel to STFT spectrogram using PyTorch.

    Args:
        signal (torch.Tensor): Input EEG signal (1D tensor).
        n_fft (int): FFT window size (default: 512).
        hop_length (int): Step size for moving window (default: 128).
        win_length (int): Window size for each FFT (default: 512).

    Returns:
        torch.Tensor: Normalized STFT spectrogram.
    """
    if not torch.is_tensor(signal):
        signal = torch.tensor(signal, dtype=torch.float32)

    # Compute STFT (PyTorch native for GPU support)
    stft_result = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length, device=signal.device),
        center=True,
        normalized=False,
        onesided=True,  # Reduces redundancy by keeping only one side of the spectrum
        return_complex=True
    )

    # Compute magnitude
    magnitude = torch.abs(stft_result)

    # Normalize: Min-max scaling
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-6)

    # Log scaling for better contrast
    magnitude = torch.log1p(magnitude)

    return magnitude


# ************************** Bandpass Filter **************************

def bandpass_filter(data, lowcut=0.5, highcut=30, fs=200, order=4):
    """
    Apply a bandpass filter to the EEG signal.

    Args:
        data (np.ndarray): 1D EEG signal.
        lowcut (float): Lower frequency bound.
        highcut (float): Upper frequency bound.
        fs (int): Sampling rate.
        order (int): Filter order.

    Returns:
        np.ndarray: Filtered EEG signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


# ************************** EEG Dataset (Updated) **************************

# class CNNDataset(Dataset):
#     CLASSES = ['Seizure', 'LPD']
#     CHANNELS = 19
#
#     def __init__(self, ids, base_path, target_path, transform=None, device="cpu"):
#         """
#         Args:
#             ids (list): List of EEG IDs.
#             base_path (str): Path to EEG `.parquet` files.
#             target_path (str): Path to CSV containing labels.
#             transform (callable, optional): Transform to apply to spectrograms.
#             device (str): Device to store tensors ("cpu" or "cuda").
#         """
#         self.id_list = ids
#         self.base_path = base_path
#         self.target_path = target_path
#         self.transform = transform
#         self.device = torch.device(device)
#
#         # Load and encode labels
#         self.df = pd.read_csv(self.target_path)
#         self.labels_df = self.df[self.df['target'].isin(self.CLASSES)]
#         self.labels_dict = dict(zip(self.labels_df['eeg_id'].astype(str),
#                                     self.labels_df['target'].map({cls: i for i, cls in enumerate(self.CLASSES)})))
#
#     def process_channel_data(self, data):
#         """Process individual channel data with bandpass filter"""
#         if isinstance(data, pd.Series):
#             data = data.values
#         return bandpass_filter(data, lowcut=0.5, highcut=30, fs=200, order=4)
#
#     def __getitem__(self, index):
#         file_id = self.id_list[index]
#         file_path = os.path.join(self.base_path, f"{file_id}.parquet")
#
#         try:
#             # Load EEG data
#             data = pd.read_parquet(file_path).drop('EKG', axis=1)
#             center = len(data) // 2
#             start, end = center - 5000, center + 5000
#
#             # Ensure sufficient data length
#             if end - start < 10000:
#                 raise ValueError(f"Insufficient data length for file {file_id}")
#
#             central_10_seconds = data.iloc[start:end, :]
#
#             # Apply bandpass filter to each channel separately
#             filtered_data = np.array(
#                 [self.process_channel_data(central_10_seconds.iloc[:, i]) for i in range(self.CHANNELS)])
#
#             # Convert to PyTorch tensor and send to correct device
#             filtered_data = torch.tensor(filtered_data, dtype=torch.float32, device=self.device)
#
#             # Compute STFT for each channel
#             cnn_data = [create_stft_spectrogram_torch(filtered_data[i]) for i in range(self.CHANNELS)]
#             cnn_tensor = torch.stack(cnn_data)  # Shape: (CHANNELS, FREQ_BINS, TIME_BINS)
#
#             if self.transform:
#                 cnn_tensor = self.transform(cnn_tensor)
#
#             label = self.labels_dict[str(file_id)]
#             return cnn_tensor, label
#
#         except Exception as e:
#             warnings.warn(f"Error processing file {file_id}: {str(e)}")
#             return None  # Skip problematic file
#
#     def __len__(self):
#         return len(self.id_list)
#
#     @classmethod
#     def get_num_classes(cls):
#         return len(cls.CLASSES)
#
#     def get_class_weights(self):
#         """
#         Compute class weights for imbalanced datasets.
#         """
#         labels = np.array([self.labels_dict[str(id)] for id in self.id_list])
#         class_counts = np.bincount(labels)
#         total_samples = len(labels)
#         weights = total_samples / (len(self.CLASSES) * class_counts)
#         return torch.FloatTensor(weights).to(self.device)


# ************************** LSTM Dataset **************************


class LSTMDataset(Dataset):
    CLASSES = ['Seizure', 'LPD']
    CHANNELS = 19

    def __init__(self, ids, base_path, target_path, window_size=200, step_size=100, transform=None):
        self.id_list = ids
        self.base_path = base_path
        self.target_path = target_path

        self.window_size = window_size
        self.step_size = step_size
        self.transform = transform

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.CLASSES)

        self.df = pd.read_csv(self.target_path)
        self.labels_df = self.df[self.df['target'].isin(self.CLASSES)]
        self.labels_dict = dict(zip(self.labels_df['eeg_id'].astype(str),
                                    self.label_encoder.transform(self.labels_df['target'])))

    def process_channel_data(self, data):
        """Process individual channel data with bandpass filter"""
        if isinstance(data, pd.Series):
            data = data.values
        return bandpass_filter(data, lowcut=0.5, highcut=30, fs=200, order=4)

    def sliding_window(self, signal):
        windows = []
        for i in range(0, len(signal) - self.window_size + 1, self.step_size):
            window = signal[i:i + self.window_size]
            windows.append(window)
        return np.array(windows)

    def __getitem__(self, index):
        file_id = self.id_list[index]
        file_path = os.path.join(self.base_path, f"{file_id}.parquet")

        try:
            # Load and preprocess data
            data = pd.read_parquet(file_path).drop('EKG', axis=1)
            center = len(data) // 2
            start, end = center - 1000, center + 1000

            # Ensure we have enough data
            if end - start < 2000:
                raise ValueError(f"Insufficient data length for file {file_id}")

            central_10_seconds = data.iloc[start:end, :]

            # Apply bandpass filter to each channel separately
            filtered_data = pd.DataFrame(
                {col: self.process_channel_data(central_10_seconds[col])
                 for col in central_10_seconds.columns},
                index=central_10_seconds.index
            )

            # Standardize the data
            scaler = StandardScaler()
            df_scaled = pd.DataFrame(
                scaler.fit_transform(filtered_data),
                columns=filtered_data.columns
            )

            # Initialize lists for processed data
            lstm_data = []
            # Process each channel group
            for i in range(self.CHANNELS):
                channel_data = df_scaled.iloc[:, i]
                windowed_data = self.sliding_window(channel_data)
                lstm_data.append(windowed_data)

            # Convert to tensors
            lstm_tensor = torch.tensor(np.stack(lstm_data, axis=0), dtype=torch.float32)

            if self.transform:
                lstm_tensor = self.transform(lstm_tensor)

            label = self.labels_dict[str(file_id)]
            return lstm_tensor, label

        except Exception as e:
            print(f"Error processing file {file_id}: {str(e)}")


    def __len__(self):
        return len(self.id_list)


    @classmethod
    def get_num_classes(cls):
        return len(cls.CLASSES)

    def get_class_weights(self):
        labels = np.array([self.labels_dict[str(id)] for id in self.id_list])
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        weights = total_samples / (len(self.CLASSES) * class_counts)
        return torch.FloatTensor(weights)
