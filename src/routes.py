from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import pandas as pd
import logging

from src.model import (
    get_model,
    get_features,
    get_cat_features
)
from src.prediction_type import MODEL, PredictionRequest

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Model"])


def check_model_loaded() -> None:
    """Проверяет, загружена ли модель. Если нет — вызывает ошибку 503."""
    if not get_model() or not get_features() or not get_cat_features():
        logger.error("Модель не загружена!")
        raise HTTPException(
            status_code=503,
            detail="Модель не загружена. Попробуйте позже."
        )


@router.get("/model_info", summary="Информация о загруженной модели")
async def get_model_info() -> Dict[str, Any]:
    """Возвращает информацию о модели (фичи, тип модели, время загрузки)."""
    check_model_loaded()

    return {
        "model_type": "CatBoost",
        "current_model": MODEL,
        "features": get_features(),
        "cat_features": get_cat_features(),
        "loaded_at": datetime.now().isoformat()
    }


@router.post("/predict", summary="Предсказание вероятности")
async def predict(request: PredictionRequest) -> Dict[str, Any]:
    """
    Предсказывает вероятность целевого события на основе входных данных.
    """

    check_model_loaded()

    try:
        # Подготавливаем данные в формате DataFrame
        input_data = {k: [v] for k, v in request.dict().items() if v is not None}
        df = pd.DataFrame(input_data)

        # Приводим категориальные фичи к правильному типу
        cat_features = get_cat_features() or []

        for feature in cat_features:
            df[feature] = df[feature].astype('category')

        # Получаем вероятность
        model = get_model()
        proba = model.predict_proba(df)[0, 1]

        return {
            "probability": float(proba),
            "prediction": int(proba > 0.5),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Ошибка предсказания: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка при предсказании: {str(e)}"
        )


@router.get("/features", summary="Уникальные значения фичей")
async def get_features_value() -> Dict[str, Any]:
    """Возвращает уникальные значения фичей для модели."""
    df = pd.read_parquet('data/datasets/cross.parquet')
    return {feature: df[feature].unique().tolist() for feature in get_cat_features()}
