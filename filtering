
from scipy.signal import butter, filtfilt


# ************************** Bandpass Filter **************************
def butter_bandpass(lowcut=0.5, highcut=40, fs=200, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, lowcut=0.5, highcut=40, fs=200, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y
