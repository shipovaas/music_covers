import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import warnings

# Игнорируем предупреждения
warnings.filterwarnings("ignore")

# Конфигурация
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 128
BATCH_SIZE = 8
NUM_IMAGES = 16  # Изменили на 16 (кратно BATCH_SIZE)

# Загрузка модели (используем ваш предыдущий код)
from cover_generator import UNet, Diffusion, EMBEDDING_DIM, FEAT_DIM, TIME_EMB_DIM, TIMESTEPS


def load_model(model_path):
    # Создаем модель с правильными размерностями
    model = UNet(EMBEDDING_DIM, FEAT_DIM).to(DEVICE)
    diffusion = Diffusion(TIMESTEPS).to(DEVICE)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model_state_dict = checkpoint['model_state_dict']

    # Модифицируем state_dict для совместимости
    new_state_dict = {}
    for key, value in model_state_dict.items():
        if key == 'cond_mlp.0.weight':
            # Изменяем размерность весов для cond_mlp.0.weight
            if value.shape[1] == 152 and model.cond_mlp[0].weight.shape[1] == 256:
                # Создаем новые веса с правильной размерностью
                new_weight = torch.zeros(128, 256)
                new_weight[:, :152] = value  # Копируем существующие веса
                new_state_dict[key] = new_weight
            else:
                new_state_dict[key] = value
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model, diffusion


def generate_batch(model, diffusion, n_images):
    embed = torch.randn(n_images, EMBEDDING_DIM).to(DEVICE)
    features = torch.randn(n_images, FEAT_DIM).to(DEVICE)

    images = diffusion.sample(
        model,
        embed=embed,
        features=features,
        n=n_images,
        img_size=IMG_SIZE,
        device=DEVICE
    )
    return images


def calculate_ssim(real_images, fake_images):
    # Обрезаем до минимального количества изображений
    min_len = min(len(real_images), len(fake_images))
    real_images = real_images[:min_len]
    fake_images = fake_images[:min_len]

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    ssim_score = ssim(fake_images, real_images)
    return ssim_score.item()


def calculate_is(fake_images):
    inception = InceptionScore(normalize=True).to(DEVICE)
    inception.update(fake_images)
    is_score = inception.compute()
    return is_score[0].item(), is_score[1].item()  # (mean, std)


def calculate_fid(real_images, fake_images):
    # Обрезаем до минимального количества изображений
    min_len = min(len(real_images), len(fake_images))
    real_images = real_images[:min_len]
    fake_images = fake_images[:min_len]

    fid = FrechetInceptionDistance(feature=64, normalize=True).to(DEVICE)

    # FID требует uint8 изображения [0-255]
    real_images = (real_images * 255).byte()
    fake_images = (fake_images * 255).byte()

    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    return fid.compute().item()


def load_real_images(data_dir, num_images):
    """Загрузка реальных изображений для сравнения"""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    images = []
    files = [f for f in os.listdir(data_dir) if f.endswith('.png')][:num_images]

    for f in files:
        img = Image.open(os.path.join(data_dir, f)).convert('RGB')
        images.append(transform(img))

    return torch.stack(images).to(DEVICE)


def main():
    MODEL_PATH = "models/diffusion_model_epoch_100.pth"
    REAL_IMAGES_DIR = "data/diffusion_data/"  # Папка с реальными изображениями


    print("Loading model...")
    model, diffusion = load_model(MODEL_PATH)


    print(f"Generating {NUM_IMAGES} images...")
    fake_images = []
    for _ in tqdm(range(0, NUM_IMAGES, BATCH_SIZE)):
        batch = generate_batch(model, diffusion, BATCH_SIZE)
        fake_images.append(batch)
    fake_images = torch.cat(fake_images, dim=0)


    print("Loading real images...")
    real_images = load_real_images(REAL_IMAGES_DIR, NUM_IMAGES)


    os.makedirs("generated_samples", exist_ok=True)
    save_image(fake_images, "generated_samples/samples.png", nrow=4, normalize=True)


    print("\nCalculating metrics...")


    ssim_score = calculate_ssim(real_images, fake_images)
    print(f"SSIM: {ssim_score:.4f} (closer to 1 is better)")


    is_mean, is_std = calculate_is(fake_images)
    print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f} (higher is better)")


    fid_score = calculate_fid(real_images, fake_images)
    print(f"FID: {fid_score:.2f} (lower is better)")


if __name__ == "__main__":
    main()
