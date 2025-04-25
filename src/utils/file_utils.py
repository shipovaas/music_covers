import json
import os


def save_tracks_to_json(tracks, file_path):
    """
    Сохраняет данные о треках в JSON-файл. Если файл уже существует, данные добавляются в конец.

    :param tracks: Список треков (словарей).
    :param file_path: Путь к JSON-файлу.
    """
    if os.path.exists(file_path):
        # Если файл существует, загружаем старые данные
        with open(file_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    # Добавляем новые треки
    existing_data.extend(tracks)

    # Сохраняем все данные обратно в файл
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)