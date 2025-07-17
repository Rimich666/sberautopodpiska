from contextlib import asynccontextmanager
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from src.prediction_type import ModelType
from src.routes import router as model_router
from src.model import ModelCache


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler для управления жизненным циклом приложения"""
    try:
        ModelCache.load_model()
        print("✅ Модель успешно загружена")
        yield
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {str(e)}")
        raise
    finally:
        # Здесь можно добавить код для очистки при завершении
        print("🛑 Приложение завершает работу")


app = FastAPI(
    title="CatBoost Prediction API",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешает все домены
    allow_credentials=True,
    allow_methods=["*"],  # Разрешает все методы
    allow_headers=["*"],  # Разрешает все заголовки
)

# Подключаем роуты
app.include_router(model_router)

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8800)
