import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.data_preparation import load_and_prepare_data


# Определение архитектуры полносвязной нейросети
class AudioNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(AudioNN, self).__init__()
        self.hidden_layer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.output_layer = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.hidden_layer(x)
        return self.output_layer(x)

    def get_hidden_representation(self, x):
        """
        Возвращает скрытые представления (выходы скрытого слоя).
        """
        return self.hidden_layer(x)


def train_model(X_train, y_train, input_size, output_size, num_epochs=20, learning_rate=0.001):
    """
    Обучает модель на тренировочных данных.

    :param X_train: torch.Tensor, тренировочные данные.
    :param y_train: torch.Tensor, метки классов для тренировочных данных.
    :param input_size: int, размер входных данных.
    :param output_size: int, количество классов.
    :param num_epochs: int, количество эпох обучения.
    :param learning_rate: float, скорость обучения.
    :return: обученная модель.
    """
    # Создание модели
    model = AudioNN(input_size, output_size)

    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Обучение модели
    for epoch in range(num_epochs):
        # Прямой проход
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Обратный проход
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model


def evaluate_model(model, X_test, y_test):
    """
    Оценивает модель на тестовых данных.

    :param model: обученная модель.
    :param X_test: torch.Tensor, тестовые данные.
    :param y_test: torch.Tensor, метки классов для тестовых данных.
    :return: float, точность модели.
    """
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_classes = torch.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test.numpy(), y_pred_classes.numpy())
    return accuracy


def save_hidden_representations(model, features, output_path):
    """
    Извлекает и сохраняет скрытые представления.

    :param model: обученная модель.
    :param features: numpy.ndarray, входные данные.
    :param output_path: str, путь для сохранения скрытых представлений.
    """
    with torch.no_grad():
        hidden_representations = model.get_hidden_representation(torch.tensor(features, dtype=torch.float32))
    np.save(output_path, hidden_representations.numpy())
    print(f"Скрытые представления сохранены в '{output_path}'")


def save_track_metadata(labels, index_to_label, track_ids, track_titles, artists, output_path):
    """
    Сохраняет метаданные треков (например, жанры, ID, названия и исполнителей).

    :param labels: list, числовые метки классов.
    :param index_to_label: dict, соответствие индексов жанрам.
    :param track_ids: list, ID треков.
    :param track_titles: list, названия треков.
    :param artists: list, исполнители треков.
    :param output_path: str, путь для сохранения метаданных.
    """
    # Преобразуем числовые метки обратно в жанры и добавляем информацию о треках
    metadata = [
        {
            "index": idx,
            "genre": index_to_label[idx],
            "track_id": track_id,
            "track_title": track_title,
            "artist": artist
        }
        for idx, track_id, track_title, artist in zip(labels, track_ids, track_titles, artists)
    ]

    # Сохраняем метаданные в JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"Список треков сохранен в '{output_path}'")

    # Сохраняем только идентификаторы треков в отдельный файл
    ids_output_path = output_path.replace("_tracks.json", "_ids.json")
    with open(ids_output_path, 'w', encoding='utf-8') as f:
        json.dump(track_ids, f, indent=4, ensure_ascii=False)
    print(f"Идентификаторы треков сохранены в '{ids_output_path}'")


def main():
    # Путь к JSON-файлу с данными
    json_path = 'data/chart_tracks_with_audio.json'

    # Загрузка данных
    features, labels, track_ids, track_titles, artists = load_and_prepare_data(json_path)

    # Преобразование меток (жанров) в числовые индексы
    unique_labels = list(set(labels))  # Уникальные жанры
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    labels = [label_to_index[label] for label in labels]

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Преобразование данных в тензоры
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Параметры модели
    input_size = X_train.shape[1]  # Количество аудиофич (например, 8)
    output_size = len(unique_labels)  # Количество жанров

    # Обучение модели
    model = train_model(X_train, y_train, input_size, output_size, num_epochs=20, learning_rate=0.001)

    # Оценка модели
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Точность на тестовой выборке: {accuracy:.4f}")

    # Сохранение скрытых представлений
    hidden_representations_path = 'hidden_representations.npy'
    save_hidden_representations(model, features, hidden_representations_path)

    # Сохранение метаданных треков
    track_metadata_path = 'hidden_representations_tracks.json'
    save_track_metadata(labels, index_to_label, track_ids, track_titles, artists, track_metadata_path)

    # Сохранение обученной модели
    model_path = 'audio_nn.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Модель сохранена в '{model_path}'")


if __name__ == '__main__':
    main()