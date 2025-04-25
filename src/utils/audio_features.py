import librosa
import numpy as np
import os
import json

def extract_audio_features(file_path):
    """
    Извлекает аудиофичи из аудиофайла.

    :param file_path: Путь к аудиофайлу.
    :return: Словарь с аудиофичами.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)  # tempo возвращается как число
        tempo = float(tempo)  # Преобразуем tempo в число (float)
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
        rmse = float(np.mean(librosa.feature.rms(y=y)))

        # Дополнительные аудиофичи
        chroma_stft = float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
        melspectrogram = float(np.mean(librosa.feature.melspectrogram(y=y, sr=sr)))
        spectral_contrast = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))

        # Собираем все аудиофичи
        audio_features = {
            "tempo": tempo,  # tempo теперь точно сохраняется как число
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "zero_crossing_rate": zero_crossing_rate,
            "rmse": rmse,
            "chroma_stft": chroma_stft,
            "melspectrogram": melspectrogram,
            "spectral_contrast": spectral_contrast,
        }

        # Нормализуем аудиофичи
        return normalize_audio_features(audio_features)

    except Exception as e:
        print(f"Ошибка при извлечении аудиофич из {file_path}: {e}")
        return {}


def normalize_audio_features(audio_features):
    """
    Нормализует аудиофичи.

    :param audio_features: Словарь с аудиофичами.
    :return: Словарь с нормализованными аудиофичами.
    """
    # Определяем минимальные и максимальные значения для нормализации
    feature_ranges = {
        "tempo": (0, 300),  # Примерный диапазон темпа (BPM)
        "spectral_centroid": (0, 8000),  # Примерный диапазон спектрального центроида
        "spectral_bandwidth": (0, 8000),  # Примерный диапазон спектральной ширины
        "zero_crossing_rate": (0, 1),  # Частота пересечения нуля (уже в диапазоне 0-1)
        "rmse": (0, 1),  # Примерный диапазон RMS энергии
        "chroma_stft": (0, 1),  # Хроматические признаки
        "melspectrogram": (0, 100),  # Примерный диапазон мел-спектрограммы
        "spectral_contrast": (0, 100),  # Примерный диапазон спектрального контраста
    }

    normalized_features = {}
    for feature, value in audio_features.items():
        if feature in feature_ranges:
            min_val, max_val = feature_ranges[feature]
            # Нормализуем значение
            normalized_features[feature] = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
        else:
            normalized_features[feature] = value  # Если диапазон не указан, оставляем как есть

    return normalized_features

def add_audio_features_to_json(json_path, audio_dir, output_path):
    """
    Добавляет аудиофичи в JSON-файл.

    :param json_path: str, путь к JSON-файлу с данными.
    :param audio_dir: str, путь к папке с аудиофайлами.
    :param output_path: str, путь для сохранения обновленного JSON-файла.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for track in data:
        audio_file = os.path.join(audio_dir, f"{track['id']}.mp3")  # Предполагаем, что имя файла = ID трека
        if os.path.exists(audio_file):
            track['audio_features'] = extract_audio_features(audio_file)
        else:
            print(f"Файл {audio_file} не найден. Пропускаем.")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Обновленный JSON сохранен в {output_path}")
