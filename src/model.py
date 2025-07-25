import glob
import json
import os
import logging
from enum import Enum
from pathlib import Path
from typing import Tuple, List, Optional
from catboost import CatBoostClassifier

from src.logger import logger
from src.prediction_type import MODEL

# # Настройка логирования
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    CATBOOST = 'catboost'
    LIGHTGBM = 'lightgbm'
    LIGHTAUTOML = 'lightautoml'
    STACKER = 'stacker'


class ModelCache:
    """Класс для кэширования модели и её параметров (реализация Singleton)."""
    _instance = None
    model = None
    features = None
    cat_features = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance

    @classmethod
    def load_model(cls) -> None:
        """Загружает последнюю модель и её параметры в кэш."""
        base_path = Path(__file__).parent
        logger.info(base_path)
        models_dir: str = base_path / MODEL
        try:
            model_path, features, cat_features = _load_latest_params(models_dir)
            logger.info(f'{os.path.abspath(models_dir)}\n')
            cls.model = CatBoostClassifier()
            cls.model.load_model(model_path)
            cls.features = features
            cls.cat_features = cat_features
            logger.info("Модель успешно загружена в кэш.")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {str(e)}")
            raise


def _load_latest_params(models_dir: str) -> Tuple[str, List[str], List[str]]:
    """Ищет и загружает последнюю версию модели и её параметров.

    Args:
        models_dir: Путь к директории с моделями и параметрами.

    Returns:
        Кортеж (путь_к_модели, список_фичей, список_категориальных_фичей).

    Raises:
        FileNotFoundError: Если нет файлов модели.
        JSONDecodeError: Если параметры повреждены.
    """
    try:
        # Ищем все файлы моделей
        model_files = glob.glob(os.path.join(models_dir, "model_*.cbm"))
        if not model_files:
            raise FileNotFoundError(f"Не найдены файлы моделей в {models_dir}")

        # Берем самую свежую модель
        latest_model = max(model_files, key=os.path.getctime)
        timestamp = '_'.join(Path(latest_model).stem.split('_')[1:])
        params_file = os.path.join(models_dir, f"params_{timestamp}.json")

        # Загружаем параметры
        with open(params_file, 'r', encoding='utf-8') as f:
            params = json.load(f)

        return latest_model, params['features'], params['cat_features']

    except json.JSONDecodeError as e:
        logger.error(f"Ошибка чтения JSON-параметров: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Ошибка загрузки параметров модели: {str(e)}")
        raise


def get_model() -> Optional[CatBoostClassifier]:
    """Возвращает закэшированную модель (если есть)."""
    return ModelCache.model if ModelCache.model else None


def get_features() -> Optional[List[str]]:
    """Возвращает список фичей (если есть)."""
    return ModelCache.features if ModelCache.features else None


def get_cat_features() -> Optional[List[str]]:
    """Возвращает список категориальных фичей (если есть)."""
    return ModelCache.cat_features if ModelCache.cat_features else None
