from src.utils.prepare_diffusion_data import prepare_diffusion_data

# Определяем пути
hidden_representations_path = 'hidden_representations.npy' 
covers_dir = 'data/covers' 
output_dir = 'data/diffusion_data' 
track_metadata_path = 'hidden_representations_tracks.json' 


print(f"Путь к скрытым представлениям: {hidden_representations_path}")
print(f"Папка с обложками: {covers_dir}")
print(f"Папка для сохранения данных: {output_dir}")
print(f"Путь к метаданным треков: {track_metadata_path}")

# Запускаем подготовку данных
prepare_diffusion_data(hidden_representations_path, covers_dir, output_dir, track_metadata_path)
