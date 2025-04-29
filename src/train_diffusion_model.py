import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from PIL import Image
from torchvision import transforms
import math
from tqdm import tqdm
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)
NUM_CORES = max(1, os.cpu_count() - 2)
NUM_WORKERS = min(4, NUM_CORES)
torch.set_num_threads(NUM_CORES)

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
IMG_SIZE = 128
EMBEDDING_DIM = 128
TIMESTEPS = 500
TIME_EMB_DIM = 128
EVAL_FREQ = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.register_buffer('emb', torch.exp(torch.arange(half_dim) * -emb))
        
    def forward(self, time):
        emb = time[:, None] * self.emb[None, :].to(time.device)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class UNet(nn.Module):
    def __init__(self, embed_dim, feat_dim):
        super().__init__()
        self.activation = nn.ReLU()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(TIME_EMB_DIM),
            nn.Linear(TIME_EMB_DIM, TIME_EMB_DIM),
            self.activation,
        )
        
        self.cond_mlp = nn.SSequential(
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

    def add_noise(self, x, t, noise=None):
        noise = noise if noise is not None else torch.randn_like(x)
        sqrt_alpha = self.sqrt_alpha_bars[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alpha_bars[t][:, None, None, None]
        return sqrt_alpha * x + sqrt_one_minus * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(0, self.timesteps, (n,))

class DiffusionDataset(Dataset):
    def __init__(self, data_dir, metadata_file):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
            if isinstance(self.metadata, list):
                self.metadata = {str(i): item for i, item in enumerate(self.metadata)}
        
        sample_feat = next(iter(self.metadata.values()))['audio_features']
        self.feat_dim = len(sample_feat) if isinstance(sample_feat, dict) else len(sample_feat)
        
        self.samples = self._build_sample_list()
        
    def _build_sample_list(self):
        valid_samples = []
        files = os.listdir(self.data_dir)
        
        for f in files:
            if not f.endswith('.npy'):
                continue
                
            try:
                num = int(''.join(filter(str.isdigit, os.path.splitext(f)[0])))
                img_file = f"image_{num}.png"
                
                if img_file in files:
                    embed = np.load(os.path.join(self.data_dir, f))
                    if isinstance(embed, np.ndarray) and embed.ndim == 1:
                        valid_samples.append((f, img_file, num))
            except:
                continue
                
        return valid_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            npy_file, img_file, num = self.samples[idx]
            
            embed = np.load(os.path.join(self.data_dir, npy_file))
            embed = torch.from_numpy(embed).float().flatten()
            if embed.shape[0] < EMBEDDING_DIM:
                embed = F.pad(embed, (0, EMBEDDING_DIM - embed.shape[0]))
            elif embed.shape[0] > EMBEDDING_DIM:
                embed = embed[:EMBEDDING_DIM]
            
            features = self.metadata[str(num)]['audio_features']
            if isinstance(ffeatures, dict):
                features = list(features.values())
            features = torch.tensor(features).float()
            
            img = Image.open(os.path.join(self.data_dir, img_file)).convert('RGB')
            img = self.transform(img)
            if img.shape != (3, IMG_SIZE, IMG_SIZE):
                raise ValueError(f"Invalid image shape: {img.shape}")
                
            return embed, features, img
            
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            return (
                torch.zeros(EMBEDDING_DIM),
                torch.zeros(self.feat_dim),
                torch.zeros(3, IMG_SIZE, IMG_SIZE)
            )

def safe_collate(batch):
    filtered_batch = []
    
    for sample in batch:
        if sample is None or len(sample) != 3:
            continue
            
        embed, features, img = sample
        
        if (isinstance(embed, torch.Tensor) and 
            isinstance(features, torch.Tensor) and 
            isinstance(img, torch.Tensor) and
            embed.dim() == 1 and
            features.dim() == 1 and
            img.dim() == 3 and
            img.shape == (3, IMG_SIZE, IMG_SIZE)):
            
            filtered_batch.append((embed, features, img))
    
    if not filtered_batch:
        return None
        
    try:
        return torch.utils.data.default_collate(filtered_batch)
    except Exception as e:
        print(f"Collate error (skipping batch): {str(e)}")
        return None

def train(data_dir, metadata_file, save_path):
    device = DEVICE
    print(f"Using device: {device} with {NUM_WORKERS} workers")
    
    print("Loading dataset...")
    dataset = DiffusionDataset(data_dir, metadata_file)
    print(f"Found {len(dataset)} valid samples")
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
        collate_fn=safe_collate,
        drop_last=True,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    
    print("Initializing model...")
    emb, feat, _ = dataset[0]
    model = UNet(emb.shape[0], feat.shape[0]).to(device)
    diffusion = Diffusion(TIMESTEPS).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            if batch is None:
                continue
                
            emb, feat, img = batch
            emb, feat, img = emb.to(device), feat.to(device), img.to(device)
            
            t = diffusion.sample_timesteps(img.size(0)).to(device)
            optimizer.zero_grad()
            
            noise = torch.randn_like(img)
            noisy, _ = diffusion.add_noise(img, t, noise)
            pred = model(noisy, t, emb, feat)
            loss = F.mse_loss(pred, noise)
            
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(loss=loss.item())

        if (epoch + 1) % EVAL_FREQ == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, f"{save_path}_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved at epoch {epoch+1}")

    torch.save(model.state_dict(), save_path)
    print("Training completed successfully!")

if __name__ == "__main__":
    train("data/diffusion_data/", "data/all_tracks.json", "diffusion_model")
