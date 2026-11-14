import torch

from pravah.ml.unet import MiniUNet


def test_unet_forward_backward_synthetic_fixture():
    model = MiniUNet(base=4)
    x = torch.rand(2, 6, 32, 32)
    prediction = model(x)
    assert tuple(prediction.shape) == (2, 2, 32, 32)
    labels = torch.randint(0, 2, (2, 32, 32))
    loss = torch.nn.functional.cross_entropy(prediction, labels)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
