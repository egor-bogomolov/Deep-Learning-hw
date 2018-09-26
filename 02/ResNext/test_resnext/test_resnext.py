import torch
from resnext.resnext import resnext50

def test_forward():
    num_classes = 20
    batch_size = 4
    tensor = torch.rand(batch_size, 3, 224, 224)
    net = resnext50(num_classes=num_classes)
    output = net(tensor)
    assert output.size() == (batch_size, num_classes)