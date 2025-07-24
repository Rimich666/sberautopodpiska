import glob
import json
from datetime import datetime
import os
from pathlib import Path
import pandas as pd
from notebooks.feature_selection import MetricNames
from notebooks.models.models import models


def load_data(model, feature_metric, part, optimisation_metric):
    piece_path = Path('hard').joinpath('sub8')
    base_path = Path(__file__).parents[2] / 'data' / 'datasets' / piece_path
    params_dir = (
            Path(__file__).parents[2] / 'data' / 'models' / 'hyper' / model / piece_path / feature_metric / str(part) / optimisation_metric)
    print(params_dir)
    print(base_path)
    train = pd.read_parquet(base_path.joinpath('cross.parquet'))
    test = pd.read_parquet(base_path.joinpath('test.parquet'))
    files = glob.glob(f"{params_dir}/best_params_*.json")

    # Выбираем самый свежий файл
    latest_file = max(files, key=os.path.getctime)
    params = json.load(open(latest_file, 'r'))

    return train, test, params


def finalize_model(final_model):
    """
    Финальная тренировка, тестирование и сохранение модели
    """
    piece_path = Path('hard').joinpath('sub8')
    base_dir = os.path.abspath('../data/models')
    model_dir = os.path.join(base_dir, f"{piece_path}")
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"model_{timestamp}.cbm")
    params_path = os.path.join(model_dir, f"params_{timestamp}.json")
    metrics_path = os.path.join(model_dir, f"metrics_{timestamp}.json")

    print(final_model.hyper_params)

    print("🔧 Финальные параметры модели:")
    for k, v in final_model.hyper_params.items():
        print(f"{k}: {v}")

    # 2. Обучение финальной модели
    print("\n🚀 Обучение финальной модели на всех тренировочных данных...")

    final_model.fit()

    # 3. Тестирование на отложенной выборке
    print("\n🧪 Тестирование на отложенной выборке...")
    # test_pred = final_model.predict_proba(X_test[features])[:, 1]
    test_auc = final_model.get_score(MetricNames.auc)

    # Дополнительные метрики
    test_report = final_model.get_report(output_dict=True)

    optimal_report = final_model.get_optimal_report(output_dict=True)
    # 4. Сохранение артефактов
    final_model.save_model(model_path)

    metadata = {
        'features': final_model.features,
        'cat_features': final_model.categorical_features,
        'params': final_model.hyper_params,
        'performance': {
            'test_auc': float(test_auc),  # Преобразуем numpy.float64
            'classification_report': test_report,
            'timestamp': timestamp
        }
    }

    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'test_auc': float(test_auc),
            'classification_report': test_report,
            'optimal_report': optimal_report
        }, f, indent=4, ensure_ascii=False)

    # 5. Отчет о результатах
    print(f"\n📊 Результаты на тестовых данных:")
    print(f"- ROC-AUC: {test_auc:.4f}")
    print("\n📝 Classification Report:")
    print(final_model.get_report())
    print(f"\n📝 Classification Report с оптимизированным порогом: {final_model.optimal_threshold}:")
    print(final_model.get_optimal_report())

    print(f"\n💾 Сохраненные артефакты:")
    print(f"- Модель: {model_path}")
    print(f"- Параметры: {params_path}")
    print(f"- Метрики: {metrics_path}")


def light_auto_ml(model=models.catboost, feature_metric=MetricNames.f1_1, part=0, optimisation_metric=MetricNames.f1_1):
    train, test, model.params = load_data(model.name, feature_metric, part, optimisation_metric)
    model.train = train
    model.test = test

    finalize_model(model)


if __name__ == "__main__":
    final_learn()
