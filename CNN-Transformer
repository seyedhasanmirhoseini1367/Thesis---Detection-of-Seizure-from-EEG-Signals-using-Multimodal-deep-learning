
# ========================== CNN-Transformer Classifier ===============================

class CNN_Transformer_Classifier(nn.Module):
    def __init__(self, in_channels=1, out_channels=[64, 64], num_classes=2, dropout=0.5, transformer_layers=2, nhead=8):
        super(CNN_Transformer_Classifier, self).__init__()
        self.l2_lambda = 0.01
        self.dropout = nn.Dropout(dropout)
        
        # CNN Feature Extractor
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[0], kernel_size=8, padding=3),
            nn.BatchNorm2d(out_channels[0]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(out_channels[0], out_channels[1], kernel_size=6, padding=2),
            nn.BatchNorm2d(out_channels[1]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Define transformer input dimension based on output channels
        d_model = sum(out_channels)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=transformer_layers
        )
        
        # Classification Layer
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )

    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        
        # Reshape to process all channels at once
        # From [batch_size, num_channels, height, width] to [batch_size * num_channels, 1, height, width]
        x_reshaped = x.view(batch_size * num_channels, 1, height, width)
        
        # Apply CNN blocks
        feat1 = self.conv_block1(x_reshaped)
        feat2 = self.conv_block2(feat1)
        
        # Pool features
        feat1_pooled = self.adaptive_pool(feat1).view(batch_size * num_channels, -1)  # [batch_size * num_channels, out_channels[0]]
        feat2_pooled = self.adaptive_pool(feat2).view(batch_size * num_channels, -1)  # [batch_size * num_channels, out_channels[1]]
        
        # Concatenate features from both blocks
        combined_features = torch.cat([feat1_pooled, feat2_pooled], dim=1)  # [batch_size * num_channels, sum(out_channels)]
        
        # Reshape back to separate batch and channel dimensions
        # [batch_size, num_channels, sum(out_channels)]
        features = combined_features.view(batch_size, num_channels, -1)
        
        # Transformer Encoder
        transformer_out = self.transformer_encoder(features)
        
        # Global average pooling over channels
        final_output = torch.mean(transformer_out, dim=1)
        
        # Classification
        output = self.classifier(final_output)
        
        return output

    def get_l2_regularization_loss(self):
        l2_loss = sum(torch.norm(param, p=2) for param in self.parameters() if param.requires_grad)
        return self.l2_lambda * l2_loss

