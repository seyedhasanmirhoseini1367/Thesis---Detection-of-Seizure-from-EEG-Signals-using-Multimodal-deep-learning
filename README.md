# Thesis---Detection-of-Seizure-from-EEG-Signals-using-Multimodal-deep-learning

Seizure detection from EEG signals is a vital task in the diagnosis and management of epilepsy. This thesis presents a multimodal deep learning framework designed to classify seizures and Lateralized Periodic Discharges (LPD) from multi-channel EEG recordings. The proposed fusion model combines both temporal and spectral information by utilizing LSTM networks to process raw temporal EEG signals and CNNs to analyze time–frequency spectrograms. These modality-specific representations are then integrated using a Transformer architecture, enabling the model to capture complex cross-modal dependencies. To evaluate the effectiveness of multimodal integration, two unimodal models (LSTM→Transformer and CNN→Transformer) are also developed and compared against the fusion model. This approach provides valuable insights into how multimodal learning can enhance the accuracy and robustness of EEG-based seizure classification.

🧪 Models
Model 1: LSTM → Transformer
Uses raw temporal EEG data to learn sequential patterns and passes the features to a Transformer for classification.
✅ Achieved 79% accuracy
<img width="619" height="1007" alt="image" src="https://github.com/user-attachments/assets/622ffde5-90cb-4757-a507-91d3b1e5b902" />

Model 2: CNN → Transformer
Applies CNN to EEG spectrograms (frequency-domain) and feeds extracted features into a Transformer.
✅ Achieved 90% accuracy
<img width="615" height="1262" alt="image" src="https://github.com/user-attachments/assets/9badb9ca-fac3-4736-8826-565a8bc140ef" />

Model 3: LSTM (temporal) + CNN (spectrogram) → Transformer
A multimodal fusion model that combines temporal and spectral features before passing them to a Transformer.
✅ Achieved 82% accuracy
<img width="849" height="947" alt="image" src="https://github.com/user-attachments/assets/6fd60009-462f-42c2-9688-3ab7aa405d8b" />
