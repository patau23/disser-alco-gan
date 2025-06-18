import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.gan import Generator, Discriminator
import os

# ========== Гиперпараметры ==========
batch_size = 64
image_size = 64
nz = 100  # размер латентного вектора
ngf = 64  # размер фич генератора
ndf = 64  # размер фич дискриминатора
num_epochs = 50
lr = 0.0002
beta1 = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Трансформации для CelebA ==========
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

# ========== Путь к изображениям ==========
# Убедись, что у тебя структура: data/celeba/faces/000001.jpg и т.д.
dataset_path = 'data/celeba'
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ========== Модели ==========
netG = Generator(nz, ngf, 3).to(device)
netD = Discriminator(3, ndf).to(device)

# ========== Функции потерь и оптимизаторы ==========
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# ========== Обучение ==========
print("🔁 Starting Training Loop...")
# ...existing code...

for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        b_size = real_images.size(0)
        real_images = real_images.to(device)

        # Метки
        real_label = torch.ones(b_size, device=device)
        fake_label = torch.zeros(b_size, device=device)

        # === Обновляем дискриминатор ===
        netD.zero_grad()

        # Потери на реальных
        output = netD(real_images)
        real_label_resized = torch.ones_like(output)  # Приводим метки к размеру выхода
        loss_real = criterion(output, real_label_resized)

        # Потери на фейках
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_images = netG(noise)
        output = netD(fake_images.detach())
        fake_label_resized = torch.zeros_like(output)  # Приводим метки к размеру выхода
        loss_fake = criterion(output, fake_label_resized)

        # Суммарная потеря дискриминатора
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizerD.step()

        # === Обновляем генератор ===
        netG.zero_grad()
        output = netD(fake_images)
        loss_G = criterion(output, real_label_resized)  # Используем метки реальных данных
        loss_G.backward()
        optimizerG.step()

    print(f"[{epoch+1}/{num_epochs}] Loss_D: {loss_D.item():.4f}  Loss_G: {loss_G.item():.4f}")

# ...existing code...
# ========== Сохраняем генератор ==========
os.makedirs('output', exist_ok=True)
torch.save(netG.state_dict(), 'output/generator.pth')
print("✅ Training complete. Model saved to output/generator.pth")
