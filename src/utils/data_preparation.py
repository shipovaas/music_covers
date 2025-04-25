import json
import numpy as np

def load_and_prepare_data(json_path, fill_missing=False):
    """
    Загружает данные из JSON-файла, фильтрует треки без аудиофичей и извлекает аудиофичи, метки, ID, названия и исполнителей.

    :param json_path: str, путь к JSON-файлу с данными.
    :param fill_missing: bool, если True, заполняет отсутствующие фичи значениями по умолчанию.
    :return: tuple (numpy.ndarray, list, list, list, list), аудиофичи, метки, ID треков, названия треков, исполнители.
    """
    # Шаг 1: Загрузка данных из JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Шаг 2: Фильтрация треков без аудиофичей
    filtered_data = [
        track for track in data if track.get('audio_features') is not None or fill_missing
    ]
    print(f"Количество треков после фильтрации: {len(filtered_data)} из {len(data)}")

    # Шаг 3: Извлечение аудиофич, меток, ID, названий и исполнителей
    features = []
    labels = []
    track_ids = []
    track_titles = []
    artists = []

    # Значения по умолчанию для аудиофич
    default_features = {
        "tempo": 0.0,
        "spectral_centroid": 0.0,
        "spectral_bandwidth": 0.0,
        "zero_crossing_rate": 0.0,
        "rmse": 0.0,
        "chroma_stft": 0.0,
        "melspectrogram": 0.0,
        "spectral_contrast": 0.0
    }

    for track in filtered_data:
        # Аудиофичи
        audio_features = track.get('audio_features', default_features)
        features.append(list(audio_features.values()))

        # Метки (жанры)
        labels.append(track.get('genre', 'Unknown'))  # Если жанр отсутствует, используем 'Unknown'

        # ID треков
        track_ids.append(track.get('id', 'Unknown'))  # Если ID отсутствует, используем 'Unknown'

        # Названия треков
        track_titles.append(track.get('title', 'Unknown'))  # Если название отсутствует, используем 'Unknown'

        # Исполнители
        artists.append(", ".join(track.get('artists', ['Unknown'])))  # Преобразуем список исполнителей в строку

    # Шаг 4: Преобразование в numpy массив
    features = np.array(features)

    return features, labels, track_ids, track_titles, artists