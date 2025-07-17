import pandas as pd
import glob
import os
import json


def explore_data_modern(df, name):
    print(f"\n🔍 Анализ датафрейма: {name}")

    # Создаем сводную таблицу с характеристиками
    analysis = pd.DataFrame({
        'Тип данных': df.dtypes,
        'Уникальных': df.nunique(),
        'Пропусков': df.isnull().sum(),
        '% Пропусков': (df.isnull().mean() * 100).round(1),
        'Пример значения': df.iloc[0] if len(df) > 0 else None
    }).sort_values('Пропусков', ascending=False)

    # Стилизация таблицы
    styled_analysis = analysis.style \
        .background_gradient(subset=['Пропусков', '% Пропусков'], cmap='Reds') \
        .format({'% Пропусков': '{:.1f}%'})

    return styled_analysis


def print_unique(df, column):
    for i, value in enumerate(df[column].unique(), 1):
        print(f"{i}. {value}")


SESSIONS_FRAME = 'sessions'
HITS_FRAME = 'hits'


def save_frame(df, frame_name, comment):
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    path = f'../data/processed/'
    filename = f'{path}{frame_name}_{timestamp}.pkl'
    df.to_pickle(filename)
    log_file = f'../data/processed/log_{frame_name}.txt'
    with open(log_file, "a", encoding="utf-8") as file:
        file.write(f"{timestamp}: {comment}\n")

    print(f'Сохранено: {filename} | Лог: {log_file}')


def load_best_params(json_path):
    """
    Загружает лучшие параметры из JSON файла
    Args:
        json_path: Полный путь к JSON файлу с параметрами
    Returns:
        dict: Словарь с параметрами модели и мета-информацией
    """
    with open(json_path, 'r') as f:
        params = json.load(f)

    return params


def load_latest_params(results_dir='../data/models/hyper'):
    """
    Загружает параметры из самого свежего JSON файла в директории
    Args:
        results_dir: Директория с файлами параметров
    Returns:
        dict: Словарь с параметрами модели и мета-информацией
    """

    # Ищем все JSON файлы с параметрами
    files = glob.glob(f"{results_dir}/best_params_*.json")

    if not files:
        raise FileNotFoundError(f"Не найдены файлы параметров в {results_dir}")

    # Выбираем самый свежий файл
    latest_file = max(files, key=os.path.getctime)
    params = json.load(open(latest_file, 'r'))
    return {
        'iterations': params['iterations'],
        'depth': params['depth'],
        'learning_rate': params['learning_rate'],
        'l2_leaf_reg': params['l2_leaf_reg'],
        'random_seed': 42,
        'task_type': 'GPU',
        'devices': '0',
        'auto_class_weights': 'Balanced',
        'verbose': 0
    }


def get_season(month):
    if month < 3 or month == 12:
        return 'winter'
    if month < 6:
        return 'spring'
    if month < 9:
        return 'summer'
    return 'autumn'


def time_of_day(hour):
    if 5 <= hour < 10:
        return 'morning'
    if 10 <= hour < 17:
        return 'day'
    if 17 <= hour < 22:
        return 'evening'
    return 'night'


def is_peak_hour(hour):
    morning_peak = 7 <= hour <= 10  # Утренний час пик 7:00-10:00
    evening_peak = 17 <= hour <= 20  # Вечерний час пик 17:00-20:00
    return 1 if morning_peak or evening_peak else 0
