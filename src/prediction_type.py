from enum import Enum
from pydantic import BaseModel, create_model
from typing import Optional, Dict, Any


class ModelType(str, Enum):
    BASE = 'base'
    KEY = 'has_keyword'
    RETURN = 'is_returning'


MODEL = ModelType.BASE


def get_request_type() -> BaseModel:
    """Создаёт PredictionRequest с разными полями в зависимости от model_type"""
    model_type = MODEL
    common_fields = {
        "utm_source": (str, ...),
        "utm_medium": (str, ...),
        "device_brand": (str, ...),
        "utm_campaign": (str, ...)
    }

    # Варианты полей для разных типов моделей
    type_specific_fields = {
        ModelType.BASE: {
            "utm_source": (str, ...),
            "utm_medium": (str, ...),
            "device_brand": (str, ...),
            "visit_number": (int, ...),
            "utm_campaign": (str, ...),
            "utm_keyword": (str, ...)
        },
        ModelType.KEY: {
            "utm_source": (str, ...),
            "utm_medium": (str, ...),
            "device_brand": (str, ...),
            "visit_number": (int, ...),
            "utm_campaign": (str, ...),
            "has_utm_keyword": (int, ...)
        },
        ModelType.RETURN: {
            "utm_source": (str, ...),
            "utm_medium": (str, ...),
            "device_brand": (str, ...),
            "is_returning": (int, ...),
            "utm_campaign": (str, ...),
            "has_utm_keyword": (int, ...)
        }
    }

    fields = {**common_fields, **type_specific_fields.get(model_type, {})}
    DynamicModel = create_model(f"PredictionRequest_{model_type}", **type_specific_fields.get(model_type))
    return DynamicModel


PredictionRequest = get_request_type()
