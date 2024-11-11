import math
import torch.nn as nn

from src.models.timm import TIMModel

class BalancedLinear(nn.Module):
  def __init__(self, in_features, num_classes):
    super(BalancedLinear, self).__init__()
    self.linear = nn.Linear(in_features, num_classes)

  def forward(self, x):
    return self.linear(x)

  def reset_parameters(self):
    nn.init.constant_(self.linear.bias, -math.log(self.linear.out_features))

class ReferralModel(TIMModel):
    def __init__(self, num_classes: int, num_channels, model_name: str, pretrained: bool, 
                 features_only: str, scriptable: bool, custom_head = False, **kwargs):
        super().__init__(model_name, num_classes, num_channels, pretrained, features_only, scriptable, **kwargs)
        
        
        if custom_head:
            in_features = self.model.get_classifier().in_features
            self.model.classifier = nn.Sequential(
                # nn.BatchNorm1d(in_features),
                nn.Linear(in_features, 1024), nn.ReLU(),
                nn.Linear(1024, 512), nn.ReLU(),
                # nn.Linear(1024, 512, bias=False), nn.ReLU(),
                # nn.BatchNorm1d(512),
                # nn.Dropout(),
                nn.Linear(512, 256), nn.ReLU(),
                # nn.BatchNorm1d(256),
                # nn.Dropout(),
                nn.Linear(256, 128), nn.ReLU(),
                # nn.BatchNorm1d(128),
                # nn.Dropout(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, num_classes)
            )

    def forward(self, x):
        return self.model(x)
