import torch
from torch.utils.data import Dataset
from utils.data_processing import load_cover_image, preprocess_cover_image


def pad_audio_features(audio_features, target_dim=13):
    """
    Дополняет аудиофичи до нужного размера, заполняя недостающие значения нулями.

    :param audio_features: Словарь с аудиофичами.
    :param target_dim: Ожидаемое количество фичей.
    :return: Список аудиофичей фиксированного размера.
    """
    feature_values = list(audio_features.values())
    if len(feature_values) < target_dim:
        feature_values.extend([0.0] * (target_dim - len(feature_values)))  # Дополняем нулями
    return feature_values[:target_dim]  # Обрезаем, если больше target_dim


class TrackDataset(Dataset):
    def __init__(self, tracks, covers_dir="data/covers", target_dim=13):
        self.tracks = tracks
        self.covers_dir = covers_dir
        self.target_dim = target_dim  # Ожидаемое количество фичей

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track = self.tracks[idx]

        # Загрузка аудиофич
        audio_features_dict = track.get("audio_features", {})
        audio_features = torch.tensor(
            pad_audio_features(audio_features_dict, target_dim=self.target_dim),
            dtype=torch.float32
        )

        # Загрузка и предобработка обложки
        cover_image = load_cover_image(track["title"], self.covers_dir)
        if cover_image:
            cover_image = preprocess_cover_image(cover_image)
        else:
            # Если обложка не найдена, возвращаем пустое изображение
            cover_image = torch.zeros((3, 256, 256))  # Черное изображение

        return audio_features, cover_image