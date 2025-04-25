import os
# import torch
from PIL import Image
import numpy as np
from train_diffusion_model import *

def generate_images_with_names(model_path, dataset, output_dir, num_samples=10):
    """
    Генерирует изображения для треков из датасета и сохраняет их с названиями треков.

    :param model_path: Путь к сохраненной модели.
    :param dataset: Объект датасета (DiffusionDataset).
    :param output_dir: Папка для сохранения сгенерированных изображений.
    :param num_samples: Количество треков для генерации.
    """
    # Определяем устройство (GPU или CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка модели
    feature_dim = len(dataset[0][1])  # Размер фич трека
    model = UNetGenerator(embedding_dim=EMBEDDING_DIM, feature_dim=feature_dim, img_size=IMG_SIZE).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Создаем папку для сохранения изображений
    os.makedirs(output_dir, exist_ok=True)

    # Генерация изображений
    with torch.no_grad():
        for i in range(num_samples):
            # Извлекаем данные из датасета
            embedding, track_features, _ = dataset[i]
            track_name = dataset.metadata[i]['title']

            # Преобразуем данные в тензоры и перемещаем на устройство
            embedding = embedding.unsqueeze(0).to(device)  # Добавляем batch dimension
            track_features = track_features.unsqueeze(0).to(device)

            # Генерируем изображение
            output = model(embedding, track_features)
            output_image = output.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Преобразуем в формат HWC
            output_image = (output_image * 255).astype(np.uint8)  # Преобразуем в диапазон [0, 255]

            # Генерируем имя файла
            sanitized_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in track_name)
            output_path = os.path.join(output_dir, f"{sanitized_name}.png")

            # Сохраняем изображение
            Image.fromarray(output_image).save(output_path)
            print(f"Сохранено: {output_path}")
if __name__ == "__main__":
    # Путь к данным
    DATA_DIR = "data/diffusion_data/"
    METADATA_FILE = "data/chart_tracks_with_audio.json"
    MODEL_SAVE_PATH = "improved_diffusion_model.pth"
    OUTPUT_DIR = "data/generated_covers/"

    # Загружаем датасет
    dataset = DiffusionDataset(DATA_DIR, METADATA_FILE)

    # Генерируем изображения для первых 10 треков
    generate_images_with_names(MODEL_SAVE_PATH, dataset, OUTPUT_DIR, num_samples=10)