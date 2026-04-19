import torch
from model import SimpleModel

def test_model_output():
    model = SimpleModel()
    x = torch.randn(1,1,256,256)
    y = model(x)

    assert y.shape == (1,1,256,256)
