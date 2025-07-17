
# ========================== Fusion Encoding =====================================

class Fusion_Classifier(nn.Module):
    def __init__(self, in_channels=1, out_channels=[64, 96], d_model=128, lstm_hidden_size=32,
                 num_classes=2, dropout=0.5, num_heads=4):
        super(Fusion_Classifier, self).__init__()

        self.l2_lambda = 0.01
        self.dropout = nn.Dropout(dropout)

        # CNN Feature Extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[0], kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels[0]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(out_channels[0], out_channels[1], kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels[1]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Adaptive Pooling for CNN Output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # LSTM for sequential feature extraction
        self.lstm = nn.LSTM(input_size=200, hidden_size=lstm_hidden_size, batch_first=True,
                          bidirectional=False)

        # Linear transformation to match d_model
        self.embedding = nn.Linear(out_channels[1] + lstm_hidden_size, d_model)

        # Multi-Head Self-Attention
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)

        # Classification Layers
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, cnnx, lstmx):
        batch_size, height, num_channels, width = cnnx.shape

        # Process all channels simultaneously through CNN
        cnnx = cnnx.view(batch_size * num_channels, 1, height, width)  # Reshape for parallel CNN
        cnnx = self.conv_layers(cnnx)
        cnnx = self.adaptive_pool(cnnx)
        cnn_out = cnnx.view(batch_size, num_channels, -1)  # Reshape back for Self-Attention

        # LSTM Processing
        batch_size, sequence_length, num_channels, input_size = lstmx.shape
        lstmx = lstmx.view(batch_size * num_channels, sequence_length, input_size)
        lstm_out, _ = self.lstm(lstmx)

        # Reshape LSTM output to match CNN output dimensions
        lstm_out = lstm_out[:, -1, :]  # Take the last output of the LSTM
        lstm_out = lstm_out.view(batch_size, num_channels, -1)  # Reshape to [batch_size, num_channels, hidden_size]

        # Now both cnn_out and lstm_out have shape [batch_size, num_channels, feature_size]
        fused_features = torch.cat((cnn_out, lstm_out), dim=-1)

        # Embedding layer to match d_model
        fused_features = self.embedding(fused_features)

        # Self-Attention
        attn_out, _ = self.self_attention(fused_features, fused_features, fused_features)
        attn_out = self.layer_norm(attn_out.mean(dim=1))  # Aggregate across channels
        attn_out = self.dropout(attn_out)

        # Classification
        output = self.classifier(attn_out)
        return output

    def get_l2_regularization_loss(self):
        l2_loss = sum(torch.norm(param, p=2) for param in self.parameters() if param.requires_grad)
        return self.l2_lambda * l2_loss

