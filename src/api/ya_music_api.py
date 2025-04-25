from yandex_music import Client
ZeIoAAG8XodFgAPFMEyPvS92EBSB2SE'):
    """
    Инициализирует клиент Яндекс Музыки.

    :param token: Токен для авторизации (если есть).
    :return: Инициализированный клиент.
    """
    if token:
        # Авторизация с токеном
        client = Client(token).init()
        print("Клиент успешно авторизован с токеном.")
    else:
        # Без авторизации
        client = Client().init()
        print("Клиент инициализирован без авторизации.")
    return client

def get_track_by_id(client, track_id):
    """
    Получает информацию о треке по его ID.

    :param client: Инициализированный клиент Яндекс Музыки.
    :param track_id: ID трека (например, '10994777:1193829').
    :return: Объект трека или None, если трек не найден.
    """
    try:
        track = client.tracks([track_id])[0]
        return track
    except Exception as e:
        print(f"Ошибка при получении трека: {e}")
        return None

def get_chart_tracks(client, chart_type='world'):
    """
    Получает список треков из чарта (например, мирового).

    :param client: Инициализированный клиент Яндекс Музыки.
    :param chart_type: Тип чарта ('world', 'russia' и т.д.).
    :return: Список треков или пустой список, если произошла ошибка.
    """
    try:
        chart_info = client.chart(chart_type)  # Получаем объект ChartInfo
        chart_tracks = chart_info.chart.tracks  # Доступ к трекам через chart.tracks
        return chart_tracks
    except Exception as e:
        print(f"Ошибка при получении чарта: {e}")
        return []


def search_tracks_with_pagination(client, query, limit=10000):
    """
    Ищет треки по запросу с использованием пагинации.

    :param client: Инициализированный клиент Яндекс Музыки.
    :param query: Поисковый запрос (например, "music", "pop").
    :param limit: Максимальное количество треков для получения.
    :return: Список треков.
    """
    tracks = []
    page = 0

    try:
        while len(tracks) < limit:
            # Выполняем поиск с учетом пагинации
            search_result = client.search(query, type_='track', page=page)
            if not search_result.tracks or not search_result.tracks.results:
                break  # Если треков больше нет, выходим из цикла

            # Добавляем треки из текущей страницы
            tracks.extend(search_result.tracks.results)

            # Увеличиваем номер страницы
            page += 1

            # Если достигли лимита, обрываем цикл
            if len(tracks) >= limit:
                break

        # Ограничиваем список до указанного лимита
        return tracks[:limit]

    except Exception as e:
        print(f"Ошибка при поиске треков: {e}")
        return []

def get_tracks_from_playlist(client, playlist_kind, user_id=None):
    """
    Получает треки из плейлиста.

    :param client: Инициализированный клиент Яндекс Музыки.
    :param playlist_kind: ID плейлиста.
    :param user_id: ID пользователя (если плейлист пользовательский).
    :return: Список треков.
    """
    try:
        if user_id:
            playlist = client.users_playlists(playlist_kind, user_id=user_id)
        else:
            playlist = client.users_playlists(playlist_kind)
        return [track.fetch_track() for track in playlist.tracks]
    except Exception as e:
        print(f"Ошибка при получении плейлиста: {e}")
        return []

def get_tracks_from_album(client, album_id):
    """
    Получает треки из альбома.

    :param client: Инициализированный клиент Яндекс Музыки.
    :param album_id: ID альбома.
    :return: Список треков.
    """
    try:
        album = client.albums_with_tracks(album_id)
        if album and album.volumes:
            return album.volumes[0]  # Возвращаем список треков из первого тома альбома
        else:
            print(f"Альбом с ID {album_id} не содержит треков.")
            return []
    except Exception as e:
        print(f"Ошибка при получении альбома: {e}")
        return []

def search_tracks(client, query, limit=100):
    """
    Ищет треки по запросу.

    :param client: Инициализированный клиент Яндекс Музыки.
    :param query: Поисковый запрос.
    :param limit: Максимальное количество треков.
    :return: Список треков.
    """
    try:
        search_result = client.search(query, type_='track', nocorrect=False)
        return search_result.tracks.results[:limit]
    except Exception as e:
        print(f"Ошибка при поиске треков: {e}")
        return []

def get_liked_tracks(client):
    """
    Получает треки, которые пользователь отметил как понравившиеся.

    :param client: Инициализированный клиент Яндекс Музыки.
    :return: Список треков.
    """
    try:
        liked_tracks = client.users_likes_tracks()
        return [track.fetch_track() for track in liked_tracks]
    except Exception as e:
        print(f"Ошибка при получении понравившихся треков: {e}")
        return []

def search_tracks_massively(client, queries, limit_per_query=100):
    """
    Выполняет массовый поиск треков по множеству запросов.

    :param client: Инициализированный клиент Яндекс Музыки.
    :param queries: Список поисковых запросов.
    :param limit_per_query: Максимальное количество треков на один запрос.
    :return: Список треков.
    """
    all_tracks = []
    for query in queries:
        try:
            search_result = client.search(query, type_='track', nocorrect=False)
            tracks = search_result.tracks.results[:limit_per_query]
            all_tracks.extend(tracks)
            print(f"Найдено {len(tracks)} треков по запросу '{query}'.")
        except Exception as e:
            print(f"Ошибка при поиске треков по запросу '{query}': {e}")
    return all_tracks

def get_all_albums_tracks(client, limit=100):
    """
    Получает треки из всех доступных альбомов.

    :param client: Инициализированный клиент Яндекс Музыки.
    :param limit: Максимальное количество альбомов.
    :return: Список треков.
    """
    all_tracks = []
    try:
        landing = client.landing('new-releases')  # Получаем новые релизы
        albums = landing.blocks[0].entities[:limit]  # Ограничиваем количество альбомов
        for album in albums:
            album_tracks = get_tracks_from_album(client, album.data.id)
            all_tracks.extend(album_tracks)
            print(f"Загружено {len(album_tracks)} треков из альбома '{album.data.title}'.")
    except Exception as e:
        print(f"Ошибка при получении альбомов: {e}")
    return all_tracks

def get_all_playlists_tracks(client, limit=100):
    """
    Получает треки из всех доступных плейлистов.

    :param client: Инициализированный клиент Яндекс Музыки.
    :param limit: Максимальное количество плейлистов.
    :return: Список треков.
    """
    all_tracks = []
    try:
        playlists = client.users_playlists_list()[:limit]  # Ограничиваем количество плейлистов
        for playlist in playlists:
            playlist_tracks = get_tracks_from_playlist(client, playlist.kind)
            all_tracks.extend(playlist_tracks)
            print(f"Загружено {len(playlist_tracks)} треков из плейлиста '{playlist.title}'.")
    except Exception as e:
        print(f"Ошибка при получении плейлистов: {e}")
    return all_tracks

def get_recommendations_tracks(client):
    """
    Получает рекомендованные треки для пользователя.

    :param client: Инициализированный клиент Яндекс Музыки.
    :return: Список треков.
    """
    all_tracks = []
    try:
        recommendations = client.feed()
        for playlist in recommendations.generated_playlists:
            playlist_tracks = [track.fetch_track() for track in playlist.tracks]
            all_tracks.extend(playlist_tracks)
            print(f"Загружено {len(playlist_tracks)} треков из рекомендаций '{playlist.title}'.")
    except Exception as e:
        print(f"Ошибка при получении рекомендаций: {e}")
    return all_tracks
from yandex_music import Client

def init_client(token='AQAAAABN