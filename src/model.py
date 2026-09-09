# CNN architecture.

import torch
import torch.nn as nn

class KeywordSpottingCNN(nn.Module):
    def __init__(self, num_classes: int = 12, in_channels: int = 1):
            super().__init__()
    
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
    
                # Block 2 
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
    
                # Block 3
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
    
                # Block 4 
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            )
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(0.3)
            self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

def count_parameters(model: nn.Module) -> int: 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__=="__main__":
    model=KeywordSpottingCNN(num_classes=12)
    dummy_input=torch.randn(4,1,40,101) # according to dataset.py's default mel config.
    output=model(dummy_input)
    print(model)
    print(f"Output Shape: {output.shape}")
    print(f"Trainable parameters: {count_parameters(model):,}")