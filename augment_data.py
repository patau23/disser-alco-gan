import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from gan_models.generator import Generator
from gan_models.discriminator import Discriminator


def train_single_gan(data_dir, device, nz=100, ngf=64, ndf=64, epochs=20, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    netG = Generator(nz, ngf, 3).to(device)
    netD = Discriminator(3, ndf).to(device)
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for _ in range(epochs):
        for real_images, _ in dataloader:
            b_size = real_images.size(0)
            real_images = real_images.to(device)
            real_label = torch.ones(b_size, device=device)
            fake_label = torch.zeros(b_size, device=device)

            netD.zero_grad()
            output = netD(real_images)
            loss_real = criterion(output, real_label)

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_images = netG(noise)
            output = netD(fake_images.detach())
            loss_fake = criterion(output, fake_label)
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizerD.step()

            netG.zero_grad()
            output = netD(fake_images)
            loss_G = criterion(output, real_label)
            loss_G.backward()
            optimizerG.step()
    return netG


def generate_images(generator, out_dir, num_images, device, nz=100):
    os.makedirs(out_dir, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        for i in range(num_images):
            noise = torch.randn(1, nz, 1, 1, device=device)
            fake = generator(noise)
            utils.save_image(fake, os.path.join(out_dir, f"{i:05d}.png"), normalize=True)


def main():
    """Augment drunk/sober dataset by training separate GANs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_root = "dataset"  # contains "sober" and "drunk" subfolders
    output_root = "augmented_dataset"
    num_images = 100

    for label in ["sober", "drunk"]:
        class_dir = os.path.join(dataset_root, label)
        if not os.path.isdir(class_dir):
            print(f"Skip {label}: directory not found")
            continue

        print(f"Training GAN for {label} images...")
        generator = train_single_gan(class_dir, device)

        out_dir = os.path.join(output_root, label)
        print(f"Generating {num_images} images for {label}...")
        generate_images(generator, out_dir, num_images, device)


if __name__ == "__main__":
    main()
