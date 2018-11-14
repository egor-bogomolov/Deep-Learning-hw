import torch.nn as nn


class DCGenerator(nn.Module):

    def __init__(self, vector_size=100, layers=128):
        super(DCGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(vector_size, layers * 8, 4, stride=1),
            nn.BatchNorm2d(layers * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(layers * 8, layers * 4, 4, stride=2),
            nn.BatchNorm2d(layers * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(layers * 4, layers * 2, 4, stride=2),
            nn.BatchNorm2d(layers * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(layers * 2, layers, 4, stride=2),
            nn.BatchNorm2d(layers),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(layers, 3, 4, stride=2),
            nn.Tanh()
        )

    def forward(self, data):
        return self.model(data)


class DCDiscriminator(nn.Module):

    def __init__(self, image_size):
        super(DCDiscriminator, self).__init__()
        nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, data):
        # TODO your code here
        pass