import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json
from PIL import Image
from torchvision import transforms, models

EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
IMG_SIZE = 128
EMBEDDING_DIM = 256
FEATURE_DIM = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiffusionDataset(Dataset):
    def __init__(self, data_dir, metadata_file):
        self.data_dir = data_dir
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])
        self.embedding_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])

        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        embedding_path = os.path.join(self.data_dir, self.embedding_files[idx])
        embedding = np.load(embedding_path)
        embedding = torch.tensor(embedding, dtype=torch.float32)

        track_metadata = self.metadata[idx]
        audio_features = track_metadata.get('audio_features', None)

        if audio_features is None:
            raise ValueError(f"Трек с индексом {idx} не содержит 'audio_features'. Проверьте данные.")

        track_features = torch.tensor(list(audio_features.values()), dtype=torch.float32)

        return embedding, track_features, image


class UNetGenerator(nn.Module):
    def __init__(self, embedding_dim, feature_dim, img_size):
        super(UNetGenerator, self).__init__()
        self.img_size = img_size

        self.fc = nn.Linear(embedding_dim + feature_dim, 512 * (img_size // 32) * (img_size // 32))

        self.encoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1), 
            nn.InstanceNorm2d(256), 
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1),  
            nn.InstanceNorm2d(128),  
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1),  
            nn.InstanceNorm2d(64),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),  
            nn.InstanceNorm2d(128), 
            nn.ReLU(),
            nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1),  
            nn.InstanceNorm2d(256), 
            nn.ReLU(),
            nn.ConvTranspose2d(256, 3, kernel_size=3, stride=1, padding=1), 
            nn.Sigmoid(),
        )

    def forward(self, embedding, track_features):
        x = torch.cat([embedding, track_features], dim=1)
        x = self.fc(x)
        x = x.view(-1, 512, self.img_size // 32, self.img_size // 32)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.mse = nn.MSELoss()

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, generated, target):
        generated = self.normalize(generated)
        target = self.normalize(target)

        gen_features = self.vgg(generated)
        target_features = self.vgg(target)

        return self.mse(gen_features, target_features)


def train_diffusion_model(data_dir, metadata_file, model_save_path):
    dataset = DiffusionDataset(data_dir, metadata_file)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    feature_dim = len(dataset[0][1])
    model = UNetGenerator(embedding_dim=EMBEDDING_DIM, feature_dim=feature_dim, img_size=IMG_SIZE).to(device)
    criterion = PerceptualLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for embeddings, track_features, images in dataloader:
            embeddings, track_features, images = embeddings.to(device), track_features.to(device), images.to(device)

            optimizer.zero_grad()
            outputs = model(embeddings, track_features)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), model_save_path)
    print(f"Модель сохранена в {model_save_path}")


if __name__ == "__main__":
    embedding = torch.randn(1, EMBEDDING_DIM) 
    track_features = torch.randn(1, FEATURE_DIM) 

    model = UNetGenerator(embedding_dim=EMBEDDING_DIM, feature_dim=FEATURE_DIM, img_size=IMG_SIZE).to(device)

    output = model(embedding.to(device), track_features.to(device))
    print(f"Размер сгенерированного изображения: {output.shape}")
