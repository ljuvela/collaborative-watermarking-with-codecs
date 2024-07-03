import torch


class WatermarkLoss(torch.nn.Module):

    def __init__(self, detector: torch.nn.Module, mode='collaborator'):
        super().__init__()
        self.detector = detector
        if mode == 'collaborator':
            self.detach_fake = False
        elif mode == 'observer':
            self.detach_fake = True
        else:
            raise ValueError('Mode must be either collaborator or observer')


    def forward(self, fake, real):

        if self.detach_fake:
            fake = fake.detach()

        d_real, d_fake = self.detector(x_real=real, x_fake=fake)

        loss_real = torch.mean((1.0 - d_real)**2)
        loss_fake = torch.mean(d_fake**2)

        loss_total = loss_real + loss_fake

        return loss_total