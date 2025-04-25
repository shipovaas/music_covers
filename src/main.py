from src.api.ya_music_api import init_client, get_chart_tracks, search_tracks_with_pagination
from src.utils.audio_downloader import download_track_preview, download_cover
from src.utils.audio_features import extract_audio_features
from src.utils.data_processing import process_tracks_data
from src.utils.file_utils import save_tracks_to_json
import os
import logging
import json

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_single_track(client, track, audio_folder, cover_folder):
    """
    Обрабатывает один трек: скачивает аудио, обложку, извлекает аудиофичи и формирует данные о треке.

    :param client: Клиент API Яндекс Музыки.
    :param track: Объект трека (Track или TrackShort).
    :param audio_folder: Папка для сохранения аудиофайлов.
    :param cover_folder: Папка для сохранения обложек.
    :return: Словарь с данными о треке или None, если произошла ошибка.
    """
    try:
        # Преобразуем TrackShort в Track, если это необходимо
        if hasattr(track, 'fetch_track'):
            track = track.fetch_track()

        # Извлекаем уникальный идентификатор трека
        track_id = str(track.id)  # Преобразуем в строку для универсальности
        file_path = os.path.join(audio_folder, f"{track_id}.mp3")
        cover_path = os.path.join(cover_folder, f"{track_id}.jpg")

        # Скачиваем аудиотрек
        if track.available:  # Проверяем, доступен ли трек для скачивания
            try:
                download_track_preview(client, track_id, file_path)
            except Exception as e:
                logger.error(f"Ошибка при скачивании трека {track_id}: {e}")
                return None
        else:
            logger.warning(f"Трек {track.title} недоступен для скачивания.")
            return None

        # Скачиваем обложку
        cover_url = track.albums[0].cover_uri if track.albums and hasattr(track.albums[0], 'cover_uri') else None
        if cover_url:
            cover_url = f"https://{cover_url.replace('%%', '400x400')}"  # Размер обложки: 400x400
            download_cover(cover_url, cover_path)

        # Извлекаем аудиофичи
        try:
            audio_features = extract_audio_features(file_path)
        except Exception as e:
            logger.error(f"Ошибка при извлечении аудиофич для трека {track_id}: {e}")
            audio_features = None

        # Формируем данные о треке
        track_data = {
            "id": track_id,  # Уникальный идентификатор трека
            "title": track.title,
            "artists": [artist.name for artist in track.artists],
            "genre": track.albums[0].genre if track.albums and hasattr(track.albums[0], 'genre') else None,
            "cover_path": cover_path,  # Путь к обложке
            "audio_features": audio_features  # Аудиофичи
        }

        return track_data

    except Exception as e:
        logger.error(f"Ошибка при обработке трека: {e}")
        return None

def process_and_save_tracks(client, chart_tracks, audio_folder, cover_folder, output_file, batch_size=100):
    """
    Обрабатывает треки из чарта и сохраняет их в JSON-файл по частям.

    :param client: Клиент API Яндекс Музыки.
    :param chart_tracks: Список треков из чарта.
    :param audio_folder: Папка для сохранения аудиофайлов.
    :param cover_folder: Папка для сохранения обложек.
    :param output_file: Путь к JSON-файлу для сохранения данных.
    :param batch_size: Количество треков в одной партии.
    """
    os.makedirs(audio_folder, exist_ok=True)
    os.makedirs(cover_folder, exist_ok=True)

    all_track_ids = []  # Список для хранения идентификаторов треков

    total_tracks = len(chart_tracks)
    for start in range(0, total_tracks, batch_size):
        end = start + batch_size
        batch = chart_tracks[start:end]
        logger.info(f"Обработка треков {start + 1}–{min(end, total_tracks)} из {total_tracks}...")

        processed_tracks = []

        for chart_track in batch:
            track_data = process_single_track(client, chart_track, audio_folder, cover_folder)
            if track_data:
                processed_tracks.append(track_data)
                all_track_ids.append(track_data["id"])  # Сохраняем только идентификатор трека

        # Сохраняем обработанные треки в файл
        save_tracks_to_json(processed_tracks, output_file)
        logger.info(f"Сохранено {len(processed_tracks)} треков в файл {output_file}")

    # Сохраняем идентификаторы треков в отдельный JSON
    with open("hidden_representations_tracks.json", "w", encoding="utf-8") as f:
        json.dump(all_track_ids, f, ensure_ascii=False, indent=4)
    logger.info(f"Сохранено {len(all_track_ids)} идентификаторов треков в hidden_representations_tracks.json")

if __name__ == "__main__":
    # Инициализируем клиент
    client = init_client()

    # Список поисковых запросов
    search_queries = ["pop", "rock", "jazz", "hip-hop", "classical", "electronic", "indie", "metal", "blues", "dance", "folk", "punk", "soul", "r&b", "country", "disco", "reggaeton", "trap", "house", "techno", "русская музыка", "английская музыка", "80s", "90s", "2000s", "2010s", "2020s", "happy", "sad", "relax", "party", "workout"]

    # Максимальное количество треков
    total_limit = 15000
    tracks = []

    # Выполняем поиск по каждому запросу
    for query in search_queries:
        if len(tracks) >= total_limit:
            break  # Если уже набрали 10 000 треков, выходим из цикла

        print(f"Ищем треки по запросу: {query}")
        query_limit = total_limit - len(tracks)  # Сколько треков еще нужно
        query_tracks = search_tracks_with_pagination(client, query=query, limit=query_limit)
        tracks.extend(query_tracks)

    # Убираем дубликаты (по ID трека)
    unique_tracks = {track.id: track for track in tracks}.values()

    # Проверяем количество уникальных треков
    unique_tracks = list(unique_tracks)
    if len(unique_tracks) < 10000:
        print(f"Недостаточно треков: найдено только {len(unique_tracks)}. Попробуйте добавить больше запросов.")
    else:
        # Ограничиваем список до 10 000 треков
        # unique_tracks = unique_tracks[:total_limit]
        print(f"Достаточно треков!!! Найдено {len(unique_tracks)} треков. Приступаем к загрузке.")

        # Папка для сохранения аудиофайлов
        audio_folder = "data/audio"

        # Папка для сохранения обложек
        cover_folder = "data/covers"

        # Файл для сохранения данных
        output_file = "data/all_tracks_with_audio.json"

        # Обрабатываем и сохраняем треки
        process_and_save_tracks(client, unique_tracks, audio_folder, cover_folder, output_file, batch_size=500)

        print(f"Сохранено {len(unique_tracks)} треков.")