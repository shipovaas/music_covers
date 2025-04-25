import os
import numpy as np
from PIL import Image
import json

def prepare_diffusion_data(hidden_representations_path, covers_dir, output_dir, track_ids_path):
    """
    Подготавливает данные для диффузионной модели, связывая скрытые представления с обложками.

    :param hidden_representations_path: str, путь к файлу с скрытыми представлениями (npy).
    :param covers_dir: str, путь к папке с изображениями обложек.
    :param output_dir: str, путь к папке для сохранения подготовленных данных.
    :param track_ids_path: str, путь к JSON-файлу с идентификаторами треков.
    """
    # Загрузка скрытых представлений
    hidden_representations = np.load(hidden_representations_path)
    print(f"Количество скрытых представлений: {len(hidden_representations)}")

    # Загрузка идентификаторов треков
    with open(track_ids_path, 'r', encoding='utf-8') as f:
        track_ids_data = json.load(f)  # Загружаем данные из JSON

    # Извлекаем только идентификаторы треков из поля "track_id"
    track_ids = [str(track["track_id"]) for track in track_ids_data]
    print(f"Количество идентификаторов треков: {len(track_ids)}")

    # Список обложек (только изображения)
    cover_files = {os.path.splitext(f)[0]: f for f in os.listdir(covers_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    print(f"Найдено {len(cover_files)} обложек.")

    # Проверка соответствия треков и обложек
    matched_tracks = []
    matched_covers = []
    for i, track_id in enumerate(track_ids):
        if track_id in cover_files:
            matched_tracks.append(hidden_representations[i])
            matched_covers.append(cover_files[track_id])
        else:
            print(f"Обложка для трека с ID {track_id} не найдена. Пропускаем.")

    # Создание выходной папки
    os.makedirs(output_dir, exist_ok=True)
    print(f"Папка для данных создана: {output_dir}")

    # Сохранение данных
    for i, (representation, cover_file) in enumerate(zip(matched_tracks, matched_covers)):
        cover_path = os.path.join(covers_dir, cover_file)
        output_image_path = os.path.join(output_dir, f'image_{i}.png')
        output_embedding_path = os.path.join(output_dir, f'embedding_{i}.npy')

        # Загрузка и сохранение изображения
        with Image.open(cover_path) as image:
            image.convert('RGB').save(output_image_path)

        # Сохранение скрытого представления
        np.save(output_embedding_path, representation)

    print(f"Данные успешно сохранены в папке: {output_dir}")