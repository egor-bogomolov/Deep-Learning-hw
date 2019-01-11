import torch
from resnext.resnext import resnext50, resnext101_32, resnext101_64, resnext152

def test_forward50():
    num_classes = 20
    batch_size = 4
    tensor = torch.rand(batch_size, 3, 224, 224)
    net = resnext50(num_classes=num_classes)
    output = net(tensor)
    assert output.size() == (batch_size, num_classes)


def test_forward101_32():
    num_classes = 20
    batch_size = 4
    tensor = torch.rand(batch_size, 3, 224, 224)
    net = resnext101_32(num_classes=num_classes)
    output = net(tensor)
    assert output.size() == (batch_size, num_classes)


def test_forward101_64():
    num_classes = 20
    batch_size = 4
    tensor = torch.rand(batch_size, 3, 224, 224)
    net = resnext101_64(num_classes=num_classes)
    output = net(tensor)
    assert output.size() == (batch_size, num_classes)


def test_forward152():
    num_classes = 20
    batch_size = 4
    tensor = torch.rand(batch_size, 3, 224, 224)
    net = resnext152(num_classes=num_classes)
    output = net(tensor)
    assert output.size() == (batch_size, num_classes)
