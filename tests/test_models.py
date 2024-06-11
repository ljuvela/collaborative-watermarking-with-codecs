import torch

def test_lfcc_lcnn():
        
    from collaborative_watermarking.models.lfcc_lcnn import LFCC_LCNN

    model = LFCC_LCNN(in_dim=1, out_dim=1, sample_rate=16000)

    batch = 2
    timesteps = 16000
    channels = 1
    x = 0.1 * torch.randn(batch, 1, timesteps, requires_grad=True)
    x = torch.nn.Parameter(x)

    scores = model.forward(x)
    # check that gradients pass
    scores.sum().backward()

    assert x.grad is not None

def test_rawnet():

    from collaborative_watermarking.models.rawnet2 import RawNet2

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    batch = 3
    timesteps = 22050
    channels = 1
    x = 0.1 * torch.randn(batch, 1, timesteps, requires_grad=True)
    x = torch.nn.Parameter(x)
    x_dev = x.to(device)

    model = RawNet2(sample_rate=16000)
    model = model.to(device)

    scores = model.forward(x_dev)
    scores.pow(2).sum().backward()

    assert x.grad is not None


