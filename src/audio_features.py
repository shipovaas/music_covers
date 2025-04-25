from utils.audio_features import add_audio_features_to_json
json_path = "data/chart_tracks_with_audio.json"  # Ваш JSON-файл с метаданными
audio_dir = "data/audio"  # Папка с аудиофайлами
output_path = "data/chart_tracks_with_audio_features.json"  # Новый JSON с аудиофичами

add_audio_features_to_json(json_path, audio_dir, output_path)