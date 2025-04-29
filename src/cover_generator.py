import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
import os
import math
from tqdm import tqdm

# Конфигурация
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 128
EMBEDDING_DIM = 256  # Обновлено до 256 в соответствии с фактическим размером эмбеддингов
FEAT_DIM = 0  # Если фичи не используются
TIMESTEPS = 500
TIME_EMB_DIM = 128
NUM_IMAGES = 756


# Класс позиционных эмбеддингов
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.register_buffer('emb', torch.exp(torch.arange(half_dim) * -emb))

    def forward(self, time):
        emb = time[:, None] * self.emb[None, :].to(time.device)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


# Модель UNet
class UNet(nn.Module):
    def __init__(self, embed_dim, feat_dim):
        super().__init__()
        self.activation = nn.ReLU()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(TIME_EMB_DIM),
            nn.Linear(TIME_EMB_DIM, TIME_EMB_DIM),
            self.activation,
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(embed_dim + feat_dim, TIME_EMB_DIM),
            self.activation,
        )

        self.down1 = self._block(3, 64)
        self.down2 = self._block(64, 128)
        self.down3 = self._block(128, 256)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            self.activation,
        )

        self.up1 = self._block(256 + 256, 128, up=True)
        self.up2 = self._block(128 + 128, 64, up=True)
        self.up3 = self._block(64 + 64, 64, up=True)

        self.emb_proj = nn.Linear(TIME_EMB_DIM, 64)
        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def _block(self, in_c, out_c, up=False):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 3, stride=2, padding=1, output_padding=1) if up
            else nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
            nn.GroupNorm(8, out_c),
            self.activation
        )

    def forward(self, x, t, embed, features):
        t_emb = self.time_mlp(t)
        cond_emb = self.cond_mlp(torch.cat([embed, features], dim=1))
        emb = self.emb_proj(t_emb + cond_emb)[:, :, None, None]

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        x = self.bottleneck(x3)

        x = self.up1(torch.cat([x, x3], dim=1))
        x = self.up2(torch.cat([x, x2], dim=1))
        x = self.up3(torch.cat([x, x1], dim=1))

        return self.final(x + emb)


# Диффузионная модель
class Diffusion(nn.Module):
    def __init__(self, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        self.register_buffers()

    def register_buffers(self):
        betas = torch.linspace(1e-4, 0.02, self.timesteps)
        alphas = 1. - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('sqrt_alpha_bars', torch.sqrt(alpha_bars))
        self.register_buffer('sqrt_one_minus_alpha_bars', torch.sqrt(1. - alpha_bars))
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)

    def sample_timesteps(self, n):
        return torch.randint(0, self.timesteps, (n,))

    @torch.no_grad()
    def sample(self, model, embed, features, n=1, img_size=64, device='cpu'):
        x = torch.randn((n, 3, img_size, img_size)).to(device)

        if embed is None:
            embed = torch.randn(n, EMBEDDING_DIM).to(device)
        if features is None:
            features = torch.randn(n, FEAT_DIM).to(device)

        for i in tqdm(reversed(range(self.timesteps)), desc='Sampling', total=self.timesteps):
            t = torch.full((n,), i, dtype=torch.long).to(device)
            pred_noise = model(x, t, embed, features)

            alpha = self.alphas[t][:, None, None, None]
            alpha_bar = self.alpha_bars[t][:, None, None, None]
            beta = self.betas[t][:, None, None, None]

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * pred_noise) + torch.sqrt(
                beta) * noise

        x = (x.clamp(-1, 1) + 1) / 2
        return x


def load_embeddings(embedding_dir, num_embeddings, expected_dim):
    embeddings = []
    for i in range(1, num_embeddings + 1):
        embedding_path = os.path.join(embedding_dir, f'embedding_{i}.npy')
        if os.path.exists(embedding_path):
            embedding = np.load(embedding_path)
            if embedding.shape[0] != expected_dim:
                print(
                    f"Warning: Embedding {embedding_path} has size {embedding.shape[0]}. Expected {expected_dim}. Resizing...")
                # Простое решение: обрезаем или дополняем нулями до нужного размера
                if embedding.shape[0] > expected_dim:
                    embedding = embedding[:expected_dim]
                else:
                    padding = np.zeros(expected_dim - embedding.shape[0])
                    embedding = np.concatenate([embedding, padding])
            embeddings.append(embedding)
        else:
            print(f"Warning: Embedding file {embedding_path} not found. Using random embedding.")
            embeddings.append(np.random.randn(expected_dim))
    return torch.tensor(np.array(embeddings), dtype=torch.float32)


def generate_images_with_embeddings(model_path, embedding_dir, output_dir='generated_images', num_images=NUM_IMAGES):
    os.makedirs(output_dir, exist_ok=True)

    # Инициализация модели
    model = UNet(EMBEDDING_DIM, FEAT_DIM).to(DEVICE)
    diffusion = Diffusion(TIMESTEPS).to(DEVICE)

    # Загрузка checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE)

    # Модифицируем state_dict для совместимости
    model_state_dict = checkpoint['model_state_dict']

    # Удаляем проблемные ключи, если они есть
    keys_to_remove = [k for k in model_state_dict.keys() if 'cond_mlp.0' in k]
    for k in keys_to_remove:
        del model_state_dict[k]

    # Загружаем с ignore_missing_keys=True
    model.load_state_dict(model_state_dict, strict=False)

    model.eval()

    print(f"Generating {num_images} images...")

    # Загрузка эмбеддингов
    embed = load_embeddings(embedding_dir, num_images, EMBEDDING_DIM).to(DEVICE)

    # Создаем фичи (если не используются, можно оставить нули)
    features = torch.zeros(num_images, FEAT_DIM).to(DEVICE)

    generated_images = diffusion.sample(
        model,
        embed=embed,
        features=features,
        n=num_images,
        img_size=IMG_SIZE,
        device=DEVICE
    )

    for i in range(num_images):
        save_image(generated_images[i], os.path.join(output_dir, f'generated_{i + 1}.png'))

    print(f"Successfully generated {num_images} images in '{output_dir}' directory")


if __name__ == "__main__":
    MODEL_PATH = "models/diffusion_model_epoch_100.pth"
    EMBEDDING_DIR = "data/diffusion_data"
    generate_images_with_embeddings(MODEL_PATH, EMBEDDING_DIR)
