import torch
from torch import nn
from einops.layers.torch import Rearrange

class CNNDecoder(nn.Module):
    def __init__(self, config):
        super(CNNDecoder, self).__init__()
        self.config = config
        w1 = self.config.dataset.image_size // self.config.dataset.patch_size
        w2 = self.config.dataset.image_size // self.config.dataset.patch_size
        self.decoder = nn.Sequential(
            Rearrange(
                    'b (w1 w2) p -> b p w1 w2', w1=w1, w2=w2
                ),
            # input is of shape 256 x 32 x 32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: 128 x 64 x 64
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 64 x 128 x 128
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: 32 x 256 x 256
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, self.config.model.kwargs.in_chans, kernel_size=4, stride=2, padding=1),  # Output: 1 x 512 x 512
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x