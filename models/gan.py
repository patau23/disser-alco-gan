import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, nz, ngf, nc):
        """
        nz: размер латентного вектора (обычно 100)
        ngf: размер фичей в генераторе
        nc: число каналов (3 для RGB)
        """
        super().__init__()
        self.main = nn.Sequential(
            # Вход: Z, выход: ngf*8 x 4 x 4
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),

            # ngf*4 x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),

            # ngf*2 x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),

            # ngf x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),

            # nc x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()  # значения от -1 до 1
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):

    def __init__(self, nc, ndf):
        """
        nc: число каналов (3 для RGB)
        ndf: размер фичей в дискриминаторе
        """
        super().__init__()
        self.main = nn.Sequential(
            # вход: nc x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.main(x)  # размер: (batch_size, 1, 1, 1)
        return out.view(-1)  # -> (batch_size,)
