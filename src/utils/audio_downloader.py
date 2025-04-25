from yandex_music import Client
import os
import requests

def download_track_preview(client, track_id, save_path):
    """
    Скачивает 30-секундный фрагмент трека.

    :param client: Инициализированный клиент Яндекс Музыки.
    :param track_id: ID трека (например, '10994777:1193829').
    :param save_path: Путь для сохранения файла.
    """
    try:
        # Получаем трек по ID
        track = client.tracks(track_id)[0]

        # Скачиваем фрагмент трека
        track.download(save_path)
        # print(f"Фрагмент трека сохранен в {save_path}")
    except Exception as e:
        print(f"Ошибка при скачивании трека {track_id}: {e}")

def download_cover(cover_url, save_path):
    """
    Скачивает обложку трека по URL и сохраняет её в указанное место.

    :param cover_url: URL обложки.
    :param save_path: Путь для сохранения обложки.
    """
    try:
        response = requests.get(cover_url, stream=True)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Обложка сохранена: {save_path}")
        else:
            print(f"Не удалось скачать обложку: {cover_url} (Статус: {response.status_code})")
    except Exception as e:
        print(f"Ошибка при скачивании обложки {cover_url}: {e}")