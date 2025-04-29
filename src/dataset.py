import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import io
import logging
import json
from tqdm import tqdm
from src.api.ya_music_api import init_client
from src.utils.audio_downloader import download_track_preview_to_memory
from src.utils.audio_features import extract_audio_features

# Настройка логгера
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def process_single_track_in_memory(track_data):
    """Обработка трека без сохранения на диск"""
    client, track = track_data
    try:
        if hasattr(track, 'fetch_track'):
            track = track.fetch_track()

        if not track.available:
            logger.warning(f"Трек {track.title} недоступен для скачивания.")
            return None

        # Скачиваем трек в память
        audio_stream = download_track_preview_to_memory(client, track.id)

        # Извлекаем аудиофичи из потока
        audio_features = extract_audio_features(audio_stream)

        return {
            "id": track.id,
            "title": track.title,
            "artists": [artist.name for artist in track.artists],
            "genre": track.albums[0].genre if track.albums and hasattr(track.albums[0], 'genre') else None,
            "audio_features": audio_features
        }
    except Exception as e:
        logger.error(f"Ошибка при обработке трека {getattr(track, 'id', 'unknown')}: {e}")
        return None

def process_and_save_tracks_in_memory(client, chart_tracks, output_file, max_workers=4, batch_size=500):
    """Параллельная обработка треков в памяти с использованием процессов"""
    processed_tracks = []

    # Используем ProcessPoolExecutor для параллельной обработки треков
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_track_in_memory, (client, track))
            for track in chart_tracks
        ]

        # Используем tqdm для отображения прогресса
        for future in tqdm(as_completed(futures), total=len(futures), desc="Обработка треков"):
            result = future.result()
            if result:
                processed_tracks.append(result)

                # Периодически сохраняем результаты
                if len(processed_tracks) % batch_size == 0:
                    save_tracks_to_json(processed_tracks, output_file)
                    logger.info(f"Сохранено {len(processed_tracks)} треков")

    # Сохраняем оставшиеся треки
    if processed_tracks:
        save_tracks_to_json(processed_tracks, output_file)

    logger.info(f"Готово! Обработано {len(processed_tracks)} треков")

def save_tracks_to_json(tracks, output_file):
    """Сохраняет треки в JSON-файл"""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tracks, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Инициализируем клиент
    client = init_client()

    # Папка с обложками
    covers_folder = "data/covers"

    # Получаем список треков из названий файлов
    track_ids = [
        os.path.splitext(filename)[0]  # Убираем расширение .jpg
        for filename in os.listdir(covers_folder)
        if filename.endswith(".jpg")
    ]

    # Получаем объекты треков по их ID
    tracks = []
    for track_id in track_ids:
        try:
            track = client.tracks([track_id])[0]  # Получаем трек по ID
            tracks.append(track)
        except Exception as e:
            logger.error(f"Ошибка при получении трека {track_id}: {e}")

    # Проверяем количество треков
    if len(tracks) == 0:
        print("Не найдено ни одного трека. Проверьте содержимое папки data/covers.")
    else:
        print(f"Найдено {len(tracks)} треков. Приступаем к обработке.")

        # Файл для сохранения данных
        output_file = "data/all_tracks.json"

        # Обрабатываем и сохраняем треки
        process_and_save_tracks_in_memory(client, tracks, output_file, max_workers=24, batch_size=500)

        print(f"Сохранено {len(tracks)} треков.")
