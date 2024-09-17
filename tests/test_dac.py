import torch
import dac

def test_forward_pass():

    sample_rate = 22050

    model = dac.model.DAC(sample_rate=sample_rate)

    batch = 2
    channels = 1
    timesteps = 20000

    x = 0.1 * torch.randn(batch, channels, timesteps)

    out = model(x, sample_rate)

    x_hat = out["audio"]
    commitment_loss = out["vq/commitment_loss"]
    codebook_loss = out["vq/codebook_loss"]

    assert x_hat.shape == x.shape


def test_backward_pass():

    sample_rate = 22050

    model = dac.model.DAC(sample_rate=sample_rate)

    batch = 2
    channels = 1
    timesteps = 20000

    x = 0.1 * torch.randn(batch, channels, timesteps)
    x = torch.nn.Parameter(x)

    out = model(x, sample_rate)

    x_hat = out["audio"]
    commitment_loss = out["vq/commitment_loss"]
    codebook_loss = out["vq/codebook_loss"]

    loss = x_hat.pow(2).mean()
    loss.backward()

    assert x.grad is not None

    assert x.grad.shape == x.shape
