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
        planes *= multiplier
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4 // multiplier, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4 // multiplier)
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

    @staticmethod
    def with_cardinality(cardinality, dim):
        # Since intermediate dimension can differ from the one in ResNet multiplier is introduced.
        return lambda *args: ResNextBlock(*args, cardinality = cardinality, multiplier=cardinality * dim // 64)


class ResNext(nn.Module):
    r"""Architecture is the same as in ResNet but with another building block.
        Thus, :class:`torchvision.models.ResNet` is used to avoid copy-pasting.
        basic_block should have cardinality as first parameter in init
    """
    def __init__(self, basic_block, cardinality, dim, layers, num_classes=1000):
        super(ResNext, self).__init__()
        self.net = models.ResNet(basic_block.with_cardinality(cardinality, dim), layers, num_classes)


    def forward(self, x):
        return self.net.forward(x)

def resnext50(pretrained=False, **kwargs):
    """Constructs a ResNeXt-50 32x4d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNext(ResNextBlock, 32, 4, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
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