import torch
from torchvision.utils import save_image
from gan_models.generator import Generator

device = 'cuda' if torch.cuda.is_available() else 'cpu'
netG = Generator(100, 64, 3).to(device)
netG.load_state_dict(torch.load('generator.pth'))
netG.eval()

with torch.no_grad():
    z = torch.randn(64, 100, 1, 1).to(device)
    fake_images = netG(z)
    save_image(fake_images, 'output/generated_faces.png', normalize=True)
