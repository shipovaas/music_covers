from src.utils.prepare_diffusion_data import prepare_diffusion_data

# Определяем пути
hidden_representations_path = 'hidden_representations.npy'  # Путь к файлу с скрытыми представлениями
covers_dir = 'data/covers'  # Папка с изображениями обложек
output_dir = 'data/diffusion_data'  # Папка для сохранения подготовленных данных
track_metadata_path = 'hidden_representations_tracks.json'  # Путь к JSON-файлу с метаданными треков


print(f"Путь к скрытым представлениям: {hidden_representations_path}")
print(f"Папка с обложками: {covers_dir}")
print(f"Папка для сохранения данных: {output_dir}")
print(f"Путь к метаданным треков: {track_metadata_path}")

# Запускаем подготовку данных
prepare_diffusion_data(hidden_representations_path, covers_dir, output_dir, track_metadata_path)