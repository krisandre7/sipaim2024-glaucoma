from torch import nn
import torch
from timm.utils.model import freeze as freeze_model
class JustExtractionModel(nn.Module):
    def __init__(self, model_path: str, num_classes: int, freeze = False,
                 custom_head = False, in_features = None, jit = False):
        super().__init__()
        self.model_path = model_path
        
        if jit:
            self.model = torch.jit.load(self.model_path).model
            in_features = self.model.classifier.in_features
        else:
            self.model = torch.load(self.model_path)['model'].model

        try:
            self.in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(self.in_features, num_classes)
            self.model = self.model.model
        except:
            self.in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(self.in_features, num_classes)
 

        if freeze:
            freeze_model(self.model)

        if custom_head:
            self.classifier = nn.Sequential(
                # nn.BatchNorm1d(in_features),
                nn.Linear(self.in_features, 1024), nn.ReLU(),
                nn.Linear(1024, 512), nn.ReLU(),
                # nn.BatchNorm1d(512),
                # nn.Dropout(0.2),
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
        else:
            self.classifier = nn.Linear(self.in_features, num_classes)
        
    def forward(self, x):
        # x = self.model.forward_features(x)
        # x = self.model.global_pool(x)
        # x = self.model.head_drop(x)
        # x = self.model.conv_head(x)
        # x = self.model.flatten(x)
        
        # return self.classifier(x)
        return self.model(x)