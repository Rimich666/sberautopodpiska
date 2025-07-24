import glob
import json
from collections import Counter
from datetime import datetime
import os
from pathlib import Path
import seaborn as sns
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Type
from catboost import Pool
from notebooks.models.catboost_model import CatBoost
from notebooks.models.lightgbm_model import LightGBM
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, auc, confusion_matrix
from sklearn.model_selection import cross_val_predict

from notebooks.feature_selection import MetricNames
from notebooks.models.helpers import find_optimal_threshold
from notebooks.models.model import Model


class FinalEnsemble:
    def __init__(self, model_classes: List[Type[Model]], use_stacking: bool = True):
        self.test = None
        self.train = None
        self.models = [model_class() for model_class in model_classes]
        self.use_stacking = use_stacking
        self.stacking_model = LogisticRegression(
            class_weight={0: 1, 1: 33},
            C=0.1,
            solver='lbfgs',
            max_iter=1000
        ) if use_stacking else None
        self.ensemble_dir = self._prepare_ensemble_dir()

    def _prepare_ensemble_dir(self) -> Path:
        """Создает директорию для сохранения ансамбля"""
        piece_path = Path('hard').joinpath('sub8')
        base_dir = Path(__file__).parents[2] / 'data' / 'models' / 'ensemble'
        ensemble_dir = base_dir / piece_path / datetime.now().strftime("%Y%m%d_%H%M%S")
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        return ensemble_dir

    def load_model_params(self, model: Model, feature_metric: str, part: int, optimisation_metric: str) -> Dict:
        """Унифицированная загрузка гиперпараметров для модели"""
        piece_path = Path('hard').joinpath('sub8')
        params_dir = (
                Path(__file__).parents[2] / 'data' / 'models' / 'hyper' /
                model.name / piece_path / feature_metric / str(part) / optimisation_metric
        )
        print(params_dir)
        files = glob.glob(f"{params_dir}/best_params_*.json")
        latest_file = max(files, key=os.path.getctime)
        params = json.load(open(latest_file, 'r'))
        return params

    def prepare_data(self, feature_metric: str, part: int, optimisation_metric: str):
        """Загрузка и подготовка данных для всех моделей"""
        piece_path = Path('hard').joinpath('sub8')
        base_path = Path(__file__).parents[2] / 'data' / 'datasets' / piece_path

        self.train = pd.read_parquet(base_path.joinpath('cross.parquet'))
        self.test = pd.read_parquet(base_path.joinpath('test.parquet'))

        for model in self.models:
            # Загрузка специфичных параметров
            model.params = {
                **self.load_model_params(model, feature_metric, part, optimisation_metric),
            }

            model.train = self.train
            model.test = self.test

    def _generate_meta_features(self):
        """Генерация мета-признаков с кросс-валидацией"""
        meta_features = np.zeros((self.train.shape[0], len(self.models)))

        for i, model in enumerate(self.models):
            try:
                meta_features[:, i] = model.get_meta()
            except Exception as e:
                print(f"Ошибка при генерации мета-признаков для модели {model.name}: {str(e)}")
                raise

        return meta_features

    def train_models(self):
        """Обучение всех моделей с их гиперпараметрами"""
        for model in self.models:
            print(f"\n🌀 Обучение {model.name}...")
            # Вызов родного fit() модели
            print(f'Гипера: {model.hyper_params}')
            model.fit()

            # Сохранение каждой модели
            model_path = self.ensemble_dir / f"{model.name}.cbm"
            model.save_model(model_path)
            print(f"💾 Модель сохранена: {model_path}")

        if self.use_stacking:
            print("\n🌀 Обучение стекинг-модели...")

            meta_train = self._generate_meta_features()

            self.stacking_model.fit(meta_train, self.train['target'])

            stacking_path = self.ensemble_dir / "stacking_model.joblib"
            joblib.dump(self.stacking_model, stacking_path)
            print(f"💾 Стекинг-модель сохранена: {stacking_path}")

    # def _detect_categorical_features(self) -> List[str]:
    #     """Автоматическое определение категориальных признаков"""
    #     return [
    #         col for col in self.train.columns
    #         if self.train[col].dtype == 'object' or self.train[col].nunique() < 10
    #     ]

    def evaluate_ensemble(self):
        """Оценка всех моделей и стекинга"""
        metrics = {}

        print('Оценка базовых моделей')
        for model in self.models:
            model_metrics, pred = self._get_model_metrics(model)
            metrics[model.name] = model_metrics
            self._print_model_metrics(model.name, model_metrics, pred)

        print('Оценка стекинга если есть')
        if self.use_stacking:
            stacking_metrics, pred = self._get_stacking_metrics()
            metrics['stacking'] = stacking_metrics
            print(stacking_metrics)
            self._print_model_metrics('Stacking', stacking_metrics, pred)

        # Сохранение всех метрик
        metrics_path = self.ensemble_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"\n💾 Все метрики сохранены: {metrics_path}")

    def _get_model_metrics(self, model: Model) -> (Dict, Dict):
        """Получение метрик для отдельной модели"""
        X_test = self.test[model.features].copy()
        X_test = model.convert_categories(X_test)
        # if model.name == 'lightgbm':
        #     for col in X_test.select_dtypes(include=['object']):
        #         X_test[col] = X_test[col].astype('category')
        y_proba = model.model.predict_proba(X_test)[:, 1]

        optimal_threshold, _ = find_optimal_threshold(self.test['target'], y_proba)

        # Самостоятельно вычисляем optimal_pred на основе порога
        optimal_pred = (y_proba >= optimal_threshold).astype(int)

        # Классификационный отчёт с дефолтным порогом (0.5)
        report_default = classification_report(
            self.test['target'],
            (y_proba >= 0.5).astype(int),
            target_names=['Class 0', 'Class 1'],
            output_dict=True
        )

        # Классификационный отчёт с оптимальным порогом
        report_optimal = classification_report(
            self.test['target'],
            optimal_pred,  # Теперь это массив 0 и 1
            target_names=['Class 0', 'Class 1'],
            output_dict=True
        )

        return {
            'auc': roc_auc_score(self.test['target'], y_proba),
            'optimal_threshold': optimal_threshold,
            'report_default': report_default,
            'report_optimal': report_optimal
        }, {
            'default': y_proba,
            'optimal': optimal_pred,
        }

    def _get_stacking_metrics(self) -> (Dict, Dict):
        """Улучшенная версия с диагностикой"""
        """ ===== ПРОВЕРОЧНЫЙ КОД ====="""
        """Проверка данных перед стекингом"""
        print("\n=== ЭКСПРЕСС-ПРОВЕРКА ДАННЫХ ===")

        # 1. Проверка целевой переменной
        print("Классы в target:",
              f"0={sum(self.test['target'] == 0)}, 1={sum(self.test['target'] == 1)}")

        # 2. Проверка фичей для каждой модели
        for model in self.models:
            X_test = self.test[model.features]
            print(f"\nМодель {model.name}:")
            print("Фичи:", X_test.shape[1], "| Примеры:", X_test.shape[0])
            print("Типы данных:", X_test.dtypes.value_counts().to_dict())

            # Проверка NaN/Inf
            print("Проблемы в данных:",
                  f"NaN={X_test.isna().sum().sum()}",
                  f"Inf={(X_test.values == np.inf).sum()}")

        # 3. Проверка стекинг-модели
        if self.use_stacking:
            print("\nСтекинг-модель:",
                  f"Готова={hasattr(self.stacking_model, 'predict_proba')}")

        """ ===== КОНЕЦ ПРОВЕРОЧНОГО КОДА ====="""
        try:
            print('Сбор предсказаний с диагностикой')
            meta_test = []
            for model in self.models:
                X_test = model.convert_categories(self.test[model.features])
                proba = model.model.predict_proba(X_test)[:, 1]
                meta_test.append(proba)

                # Вывод статистики по каждой модели
                print(f"\n🔍 {model.name}:")
                print(f"Predictions min={proba.min():.3f} max={proba.max():.3f}")
                print(f"Mean={proba.mean():.3f} | Std={proba.std():.3f}")
                print(f"Кол-во 0={(proba < 0.5).sum()} | Кол-во 1={(proba >= 0.5).sum()}")

            # 2. Проверка объединённых мета-признаков
            meta_test = np.column_stack(meta_test)
            print("\n🔥 Итоговые мета-признаки:")
            print(f"Размер: {meta_test.shape} (примеры × модели)")
            print("Пример первых 3 строк:")
            print(pd.DataFrame(meta_test).head(3).to_string())

            # 3. Проверка на "мусор"
            print("\n🧹 Проверка проблем:")
            print(f"NaN: {np.isnan(meta_test).sum()}")
            print(f"Inf: {np.isinf(meta_test).sum()}")
            print(f"Все нули: {(meta_test == 0).all(axis=0).sum()} колонок")

            print('Предсказание стекинг-модели')
            y_proba = self.stacking_model.predict_proba(meta_test)[:, 1]
            if np.all(y_proba == 0):
                raise ValueError("Стекинг-модель вернула все нули")
            print("\n=== ЭКСПРЕСС-ПРОВЕРКА ДАННЫХ ===")

            print('Расчет метрик')
            y_pred_standard = (y_proba >= 0.5).astype(int)
            optimal_threshold, _ = find_optimal_threshold(self.test['target'], y_proba)
            y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
            print(type(optimal_threshold))
            return {
                'auc': float(roc_auc_score(self.test['target'], y_proba)),
                'report': classification_report(
                    self.test['target'],
                    y_pred_standard,
                    target_names=['Class 0', 'Class 1'],
                    output_dict=True
                ),
                'optimal_threshold': float(optimal_threshold),
                'optimal_report': classification_report(
                    self.test['target'],
                    y_pred_optimal,
                    target_names=['Class 0', 'Class 1'],
                    output_dict=True
                ),
                'proba_stats': {
                    'min': float(y_proba.min()),
                    'max': float(y_proba.max()),
                    'mean': float(y_proba.mean()),
                    'std': float(y_proba.std())
                }
            }, {
                'default': y_proba,
                'optimal': y_pred_optimal,
            }

        except Exception as e:
            print(f"🔥 Критическая ошибка стекинга: {str(e)}")
            return {
                'auc': 0,
                'error': str(e),
                'proba_stats': None
            }

    def _print_model_metrics(self, name: str, metrics: Dict, pred: Dict):
        """Вывод метрик модели"""
        print(f"\n🧪 Тестирование {name}:")
        print(f"📊 ROC-AUC: {metrics['auc']:.4f}")
        print(f"📌 Оптимальный порог: {metrics['optimal_threshold']:.4f}")

        print("\n📝 Classification Report (порог 0.5):")
        print(classification_report(
            self.test['target'],
            (pred['default'] >= 0.5).astype(int),
            target_names=['Class 0', 'Class 1']
        ))

        print("\n📝 Classification Report (оптимальный порог):")
        print(classification_report(
            self.test['target'],
            pred['optimal'],
            target_names=['Class 0', 'Class 1']
        ))


def final_ensemble_learn(use_stacking=True):
    print('Вариант 1')
    ensemble = FinalEnsemble(
        model_classes=[CatBoost, LightGBM],
        use_stacking=use_stacking
    )
    ensemble.prepare_data(
        feature_metric=MetricNames.f1_1,
        part=0,
        optimisation_metric=MetricNames.f1_1
    )
    ensemble.train_models()
    ensemble.evaluate_ensemble()


if __name__ == "__main__":
    final_ensemble_learn()
