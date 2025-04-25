from src.api.ya_music_api import init_client, get_chart_tracks
from src.utils.audio_downloader import download_track_preview, download_cover
from src.utils.audio_features import extract_audio_features
from src.utils.data_processing import process_tracks_data
from src.utils.file_utils import save_tracks_to_json
import os


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

    total_tracks = len(chart_tracks)
    for start in range(0, total_tracks, batch_size):
        end = start + batch_size
        batch = chart_tracks[start:end]
        print(f"Обработка треков {start + 1}–{min(end, total_tracks)} из {total_tracks}...")

        processed_tracks = []

        for chart_track in batch:
            track = chart_track.track
            track_id = f"{track.id}:{track.albums[0].id}" 
            file_path = f"{audio_folder}/{track.title}.mp3"
            if track.available:
                download_track_preview(client, track_id, file_path)
            else:
                print(f"Трек {track.title} недоступен для скачивания.")
                continue

            cover_url = track.albums[0].cover_uri if track.albums and hasattr(track.albums[0], 'cover_uri') else None
            if cover_url:
                cover_url = f"https://{cover_url.replace('%%', '400x400')}"
                cover_path = f"{cover_folder}/{track.title}.jpg"
                download_cover(cover_url, cover_path)
            else:
                print(f"Обложка для трека {track.title} недоступна.")
                cover_path = None

            if os.path.exists(file_path):
                audio_features = extract_audio_features(file_path)
            else:
                print(f"Файл {file_path} не найден. Пропускаем.")
                audio_features = None

            track_data = {
                "title": track.title,
                "artists": [artist.name for artist in track.artists],
                "album": track.albums[0].title if track.albums else None,
                "duration_seconds": track.duration_ms // 1000,
                "genre": track.albums[0].genre if track.albums and hasattr(track.albums[0], 'genre') else None,
                "release_date": track.albums[0].release_date if track.albums and hasattr(track.albums[0], 'release_date') else None,
                "cover_path": cover_path, 
                "audio_features": audio_features
            }

            processed_tracks.append(track_data)

        processed_tracks = process_tracks_data(processed_tracks)

        save_tracks_to_json(processed_tracks, output_file)
        print(f"Сохранено {len(processed_tracks)} треков в файл {output_file}")



client = init_client()

chart_tracks = get_chart_tracks(client, chart_type='world')

audio_folder = "data/audio"

cover_folder = "data/covers"

output_file = "data/chart_tracks_with_audio.json"

process_and_save_tracks(client, chart_tracks, audio_folder, cover_folder, output_file, batch_size=100)
