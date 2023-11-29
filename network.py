import torch
from torch import nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, init_channels=16, dropout=0.5, n_conv_layers=3, norm='instance'):
        super().__init__()
        
        assert norm in ['instance', 'batch', 'none'], f'Expected norm to be in [instance, batch, none], got {norm}'

        # Initial layer will not downsample image [100, 100] --> [100, 100]
        layers = []
        init_layer = nn.ModuleDict({
                            'conv': nn.Conv2d(in_channels, init_channels, kernel_size=3),
                            'batch_norm': nn.BatchNorm2d(num_features=init_channels),
                            'instance_norm': nn.InstanceNorm2d(num_features=init_channels)
                            })
        layers.append(init_layer)
        for i in range(1, n_conv_layers):
            this_layer = nn.ModuleDict({
                            'conv': nn.Conv2d(init_channels*i, init_channels*(i+1), kernel_size=3),
                            'batch_norm': nn.BatchNorm2d(num_features=init_channels*(i+1)),
                            'instance_norm': nn.InstanceNorm2d(num_features=init_channels)
                            })
            layers.append(this_layer)

        self.norm = norm
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4800, num_classes)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer['conv'](x)

            if   self.norm == 'batch'   : x = layer['batch_norm'](x)
            elif self.norm == 'instance': x = layer['instance_norm'](x)

            x = F.relu(F.max_pool2d(x, kernel_size=(2, 2)))
            x = self.dropout(x)

        return F.softmax(self.head(x), dim=1)