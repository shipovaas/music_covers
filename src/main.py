from src.api.ya_music_api import init_client, get_chart_tracks, search_tracks_with_pagination
from src.utils.audio_downloader import download_track_preview, download_cover
from src.utils.audio_features import extract_audio_features
from src.utils.data_processing import process_tracks_data
from src.utils.file_utils import save_tracks_to_json
import os
import logging
import json

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
        if hasattr(track, 'fetch_track'):
            track = track.fetch_track()

        track_id = str(track.id) 
        file_path = os.path.join(audio_folder, f"{track_id}.mp3")
        cover_path = os.path.join(cover_folder, f"{track_id}.jpg")

        if track.available:
            try:
                download_track_preview(client, track_id, file_path)
            except Exception as e:
                logger.error(f"Ошибка при скачивании трека {track_id}: {e}")
                return None
        else:
            logger.warning(f"Трек {track.title} недоступен для скачивания.")
            return None

        cover_url = track.albums[0].cover_uri if track.albums and hasattr(track.albums[0], 'cover_uri') else None
        if cover_url:
            cover_url = f"https://{cover_url.replace('%%', '400x400')}"  # Размер обложки: 400x400
            download_cover(cover_url, cover_path)

        try:
            audio_features = extract_audio_features(file_path)
        except Exception as e:
            logger.error(f"Ошибка при извлечении аудиофич для трека {track_id}: {e}")
            audio_features = None

        track_data = {
            "id": track_id,
            "title": track.title,
            "artists": [artist.name for artist in track.artists],
            "genre": track.albums[0].genre if track.albums and hasattr(track.albums[0], 'genre') else None,
            "cover_path": cover_path,
            "audio_features": audio_features 
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

    all_track_ids = []  

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
                all_track_ids.append(track_data["id"])  

        save_tracks_to_json(processed_tracks, output_file)
        logger.info(f"Сохранено {len(processed_tracks)} треков в файл {output_file}")

    with open("hidden_representations_tracks.json", "w", encoding="utf-8") as f:
        json.dump(all_track_ids, f, ensure_ascii=False, indent=4)
    logger.info(f"Сохранено {len(all_track_ids)} идентификаторов треков в hidden_representations_tracks.json")

if __name__ == "__main__":
    client = init_client()

    search_queries = ["pop", "rock", "jazz", "hip-hop", "classical", "electronic", "indie", "metal", "blues", "dance", "folk", "punk", "soul", "r&b", "country", "disco", "reggaeton", "trap", "house", "techno", "русская музыка", "английская музыка", "80s", "90s", "2000s", "2010s", "2020s", "happy", "sad", "relax", "party", "workout"]

    total_limit = 15000
    tracks = []

    for query in search_queries:
        if len(tracks) >= total_limit:
            break 

        print(f"Ищем треки по запросу: {query}")
        query_limit = total_limit - len(tracks) 
        query_tracks = search_tracks_with_pagination(client, query=query, limit=query_limit)
        tracks.extend(query_tracks)

    unique_tracks = {track.id: track for track in tracks}.values()

    unique_tracks = list(unique_tracks)
    if len(unique_tracks) < 10000:
        print(f"Недостаточно треков: найдено только {len(unique_tracks)}. Попробуйте добавить больше запросов.")
    else:
        print(f"Достаточно треков!!! Найдено {len(unique_tracks)} треков. Приступаем к загрузке.")

        audio_folder = "data/audio"

        cover_folder = "data/covers"


        output_file = "data/all_tracks_with_audio.json"


        process_and_save_tracks(client, unique_tracks, audio_folder, cover_folder, output_file, batch_size=500)

        print(f"Сохранено {len(unique_tracks)} треков.")
