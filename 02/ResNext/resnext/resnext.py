import math

from torch import nn
from torchvision import models
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnext101-64x4': 'https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/FBResNeXt/X-101-64x4d.pkl',
    'resnext101-32x8': 'https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/20171220/X-101-32x8d.pkl',
    'resnext152-32x8': 'https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/25093814/X-152-32x8d-IN5k.pkl'
}

class ResNextBlock(nn.Module):

    r"""Class is the same with :class:`torchvision.models.Bottleneck` except for the `groups` parameter at layer conv2.
    """

    # field required by :class:`torchvision.models.ResNet` API
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, multiplier=1):
        super(ResNextBlock, self).__init__()
        self.cardinality = cardinality
        self.multiplier = multiplier
        planes *= self.multiplier
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, groups=self.cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4 // self.multiplier, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4 // self.multiplier)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNext(nn.Module):

    def __init__(self, block, cardinality, dims, layers, num_classes=1000):
        self.cardinality = cardinality
        self.dims = dims
        self.inplanes = 64
        super(ResNext, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, cardinality=self.cardinality, multiplier=self.dims))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality=self.cardinality, multiplier=self.dims))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnext50(pretrained=False, **kwargs):
    """Constructs a ResNeXt-50 32x4d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNext(ResNextBlock, 32, 4, [3, 4, 6, 3], **kwargs)
    if pretrained:
        raise NotImplementedError('Weights not found')
        # model.load_state_dict(model_zoo.load_url(model_urls['resnext50']))
    return model

def resnext101_32(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 32x8d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNext(ResNextBlock, 32, 8, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnext101-32x8']))
    return model

def resnext101_64(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 64x4d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNext(ResNextBlock, 64, 4, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnext101-64x4']))
    return model

def resnext152(pretrained=False, **kwargs):
    """Constructs a ResNeXt-152 32x8d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNext(ResNextBlock, 32, 8, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnext152-32x8']))
    return model