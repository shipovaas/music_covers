from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime
from PIL import Image
import os
from torchvision import transforms


def convert_duration_ms_to_min_sec(duration_ms):
    """
    Преобразует длительность трека из миллисекунд в минуты и секунды.

    :param duration_ms: Длительность трека в миллисекундах.9ш
    :return: Строка в формате "мин:сек".
    """
    minutes = duration_ms // 60000
    seconds = (duration_ms % 60000) // 1000
    return f"{minutes}:{seconds:02d}"


def simplify_release_date(release_date):
    """
    Преобразует дату выпуска в упрощённый формат (YYYY-MM-DD).

    :param release_date: Дата выпуска в формате ISO 8601.
    :return: Дата в формате YYYY-MM-DD или None, если дата отсутствует.
    """
    if release_date:
        try:
            return datetime.fromisoformat(release_date).date().isoformat()
        except ValueError:
            print(f"Неверный формат даты: {release_date}")
            return None
    return None


def normalize_numeric_features(tracks, feature_keys):
    """
    Нормализует числовые данные в треках.

    :param tracks: Список треков (словарей).
    :param feature_keys: Список ключей, которые нужно нормализовать.
    :return: Список треков с нормализованными значениями.
    """
    # Извлекаем значения для нормализации
    feature_values = {key: [] for key in feature_keys}
    for track in tracks:
        for key in feature_keys:
            value = track.get(key, 0)  # Если значение отсутствует, используем 0
            feature_values[key].append(value if value is not None else 0)  # Заменяем None на 0

    # Нормализуем каждую фичу
    scaler = MinMaxScaler()
    for key in feature_keys:
        values = np.array(feature_values[key]).reshape(-1, 1)
        # Проверяем, есть ли только NaN в данных
        if np.isnan(values).all():
            print(f"Все значения для {key} равны NaN. Пропускаем нормализацию.")
            continue
        # Заменяем NaN на 0 перед нормализацией
        values = np.nan_to_num(values, nan=0.0)
        normalized_values = scaler.fit_transform(values).flatten()
        for i, track in enumerate(tracks):
            track[key] = normalized_values[i]

    return tracks


def process_tracks_data(tracks):
    """
    Обрабатывает данные о треках: нормализует числовые данные и упрощает формат даты.

    :param tracks: Список треков (словарей).
    :return: Обработанный список треков.
    """
    # Упрощаем формат даты
    for track in tracks:
        track["release_date"] = simplify_release_date(track.get("release_date"))

    # Список ключей для нормализации
    feature_keys = ["duration_seconds"]

    # Нормализуем числовые данные
    processed_tracks = normalize_numeric_features(tracks, feature_keys)

    return processed_tracks


def load_cover_image(track_title, covers_dir="data/covers"):
    """
    Загружает обложку трека по названию.

    :param track_title: Название трека.
    :param covers_dir: Путь к папке с обложками.
    :return: Объект PIL.Image или None, если файл не найден.
    """
    # Формируем путь к файлу
    filename = f"{track_title}.jpg"
    filepath = os.path.join(covers_dir, filename)

    if os.path.exists(filepath):
        return Image.open(filepath).convert("RGB")  # Конвертируем в RGB
    else:
        print(f"Обложка для трека '{track_title}' не найдена в {covers_dir}.")
        return None


def preprocess_cover_image(image, size=(256, 256)):
    """
    Предобрабатывает обложку: изменяет размер и нормализует пиксели.

    :param image: Объект PIL.Image.
    :param size: Размер изображения (ширина, высота).
    :return: Тензор изображения.
    """
    transform = transforms.Compose([
        transforms.Resize(size),  # Изменяем размер
        transforms.ToTensor(),  # Преобразуем в тензор
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Нормализация в диапазон [-1, 1]
    ])
    return transform(image)