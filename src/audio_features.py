from utils.audio_features import add_audio_features_to_json
json_path = "data/chart_tracks_with_audio.json"
audio_dir = "data/audio" 
output_path = "data/chart_tracks_with_audio_features.json"

add_audio_features_to_json(json_path, audio_dir, output_path)
