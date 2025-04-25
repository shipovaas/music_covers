import json

metadata_file = 'data/chart_tracks_with_audio.json'

# Загрузка данных
with open(metadata_file, 'r') as f:
    data = json.load(f)

# Фильтруем треки, у которых 'audio_features' не равно None
filtered_data = [track for track in data if track.get('audio_features') is not None]

# Сохраняем отфильтрованные данные
filtered_metadata_file = 'data/chart_tracks_with_audio_filtered.json'
with open(filtered_metadata_file, 'w') as f:
    json.dump(filtered_data, f, indent=4, ensure_ascii=False)

print(f"Количество треков до фильтрации: {len(data)}")
print(f"Количество треков после фильтрации: {len(filtered_data)}")