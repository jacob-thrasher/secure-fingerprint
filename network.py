import torch
from torch import nn
import torch.nn.functional as F


##############
# CLASSIFIER #
##############

class SimpleCNN(torch.nn.Module):

   def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(128 * 16 * 16, 1024)
        self.fc2 = torch.nn.Linear(1024, 2)  # Assuming 601 output classes
        self.batch_norm1 = torch.nn.BatchNorm2d(32)
        self.batch_norm2 = torch.nn.BatchNorm2d(64)
        self.batch_norm3 = torch.nn.BatchNorm2d(128)

   def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.batch_norm3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(-1, 128 * 16 * 16)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

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
    

#############
#    VAE    #
#############

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, z_dim):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self.flatten = nn.Flatten()
        self.mu_proj = nn.Linear(18432, z_dim)
        self.logvar_proj = nn.Linear(18432, z_dim)
        
    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        x = self._residual_stack(x)

        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = x.view(x.size(0), -1)
        return self.mu_proj(x), self.logvar_proj(x)
        
class Decoder(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, z_dim):
        super(Decoder, self).__init__()

        self.proj = nn.Linear(z_dim, 18432)

        self._conv_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=5,
                                 stride=2, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens//2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2,
                                                out_channels=1,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, x):
        x = self.proj(x)
        x = x.view(x.size(0), 128, 12, 12) # unflatten batch of feature vectors to a batch of multi-channel feature maps

        x = self._conv_1(x)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)
    
class VAE(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, z_dim, decay=0):
        super(VAE, self).__init__()

        self._encoder = Encoder(1, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens,
                                z_dim)

        self._decoder = Decoder(num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens,
                                z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self._encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self._decoder(z)

        return x_recon, mu, logvar