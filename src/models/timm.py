import torch.nn as nn
import timm

class TIMModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, num_channels, 
                 pretrained: bool, features_only: str, 
                 scriptable: bool, **kwargs):
        super().__init__()
        self.model = timm.create_model(model_name, num_classes=num_classes, 
                                       in_chans=num_channels, pretrained=pretrained, 
                                       features_only=features_only,
                                       scriptable=scriptable,
                                       **kwargs)
    
    def forward(self, x):
        return self.model(x)