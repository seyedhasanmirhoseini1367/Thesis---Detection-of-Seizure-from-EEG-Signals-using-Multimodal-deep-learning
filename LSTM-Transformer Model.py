import torch
import torch.nn as nn

class LSTMTransformerClassifier(nn.Module):
    def __init__(self,
                 input_size,
                 lstm_hidden_size=16, 
                 lstm_layers=1,
                 transformer_heads=4,
                 transformer_layers=2,
                 dropout_rate=0.5,
                 num_classes=2): 
        super(LSTMTransformerClassifier, self).__init__()

        # Adjust hidden size dynamically to be divisible by transformer_heads
        if lstm_hidden_size % transformer_heads != 0:
            lstm_hidden_size = (lstm_hidden_size // transformer_heads + 1) * transformer_heads

        self.lstm_hidden_size = lstm_hidden_size

        self.lstm = nn.LSTM(
            input_size,
            lstm_hidden_size,
            lstm_layers,
            bidirectional=False,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )

        d_model = lstm_hidden_size * 2  

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_heads,
            dropout=dropout_rate,
            dim_feedforward=lstm_hidden_size * 4,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.dropout = nn.Dropout(dropout_rate)

        # Classification head with 2 layers: d_model -> d_model//2 -> num_classes
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, num_classes)
        )

        # Add positional encoding if needed, else comment out
        self.positional_encoding = nn.Identity()  # Placeholder, replace if you have a positional encoding implemented

    def forward(self, x):
        """
        Forward pass through the LSTM-Transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        batch_size, num_channels, seq_len = x.shape  # (B, C, T)

        lstm_outputs = []
        for i in range(num_channels):
            # Extract data for the i-th channel
            channel_data = x[:, i, :].unsqueeze(-1).contiguous()  # (batch_size, seq_len, 1)

            # Initialize LSTM hidden and cell states
            h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=x.device).contiguous()
            c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=x.device).contiguous()

            # Pass channel data through LSTM
            lstm_out, _ = self.lstm(channel_data, (h0, c0))  # (batch_size, seq_len, lstm_hidden_size)
            lstm_outputs.append(lstm_out[:, -1, :].contiguous())  # Take last time step output

        # Stack outputs from all channels along new dimension
        stacked_outputs = torch.stack(lstm_outputs, dim=1).contiguous()  # (batch_size, num_channels, lstm_hidden_size)

        encoded_input = torch.cat([stacked_outputs, stacked_outputs], dim=-1) 

        # Apply positional encoding (identity here)
        encoded_input = self.positional_encoding(encoded_input)

        # Pass through transformer encoder
        transformer_out = self.transformer_encoder(encoded_input) 

        # Extract last channel output
        out = transformer_out[:, -1, :]  # (B, d_model)

        # Dropout + classification head
        out = self.dropout(out)
        return self.classifier(out)
