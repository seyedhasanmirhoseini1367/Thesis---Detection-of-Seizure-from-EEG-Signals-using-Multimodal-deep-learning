
def create_stft_spectrogram(signal: np.ndarray, sampling_rate: int = 200, nperseg: int = 64,
                            noverlap: int = None) -> tuple:
    """
    Convert single EEG channel to STFT spectrogram.

    Args:
        signal: Input EEG signal (1D array).
        sampling_rate: Sampling rate in Hz (default: 200 Hz).
        nperseg: Length of each segment (default: 256).
        noverlap: Number of points to overlap between segments (default: nperseg // 2).

    Returns:
        tuple: (frequencies, times, spectrogram) where:
               - frequencies is the array of analyzed frequencies.
               - times is the array of time points.
               - spectrogram is the STFT matrix.

    Raises:
        ValueError: If the input signal is too short or invalid.
    """
    # Validate input signal
    if len(signal) < nperseg:
        raise ValueError("Input signal is too short for STFT computation.")

    # Default overlap is 50% of nperseg
    if noverlap is None:
        noverlap = nperseg // 2

    # Compute STFT
    frequencies, times, spectrogram = stft(signal, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)

    # Take the magnitude of the STFT
    spectrogram = np.abs(spectrogram)

    # Normalize spectrogram (optional)
    spectrogram = (spectrogram - np.min(spectrogram)) / (
            np.max(spectrogram) - np.min(spectrogram))  # Min-max normalization

    # Apply logarithmic scaling (optional)
    spectrogram = np.log1p(spectrogram)  # log(1 + x) to avoid log(0)

    return spectrogram
