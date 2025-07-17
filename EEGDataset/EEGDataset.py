import os
import pywt
import torch
import numpy as np
import pandas as pd
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


# ************************** LSTM Dataset **************************


class LSTMDataset(Dataset):
    CLASSES = ['Seizure', 'LPD']
    CHANNELS = 19

    def __init__(self, ids, base_path, target_path, window_size=200, step_size=50, transform=None):
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
