import glob
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, precision_recall_curve
import joblib
import seaborn as sns
from notebooks.feature_selection import MetricNames
from notebooks.models.catboost_model import CatBoost
from notebooks.models.lightgbm_model import LightGBM
from collections import Counter


def load_model_params(model_name: str, feature_metric: str, part: int, optimisation_metric: str) -> Dict:
    """Загрузка гиперпараметров для моделей из файлов (как в ensemble.py)"""
    piece_path = Path('hard').joinpath('sub8')
    params_dir = (
            Path(__file__).parents[2] / 'data' / 'models' / 'hyper' /
            model_name / piece_path / feature_metric / str(part) / optimisation_metric
    )
    files = glob.glob(f"{params_dir}/best_params_*.json")
    latest_file = max(files, key=os.path.getctime)
    return json.load(open(latest_file, 'r'))


class LightAutoMLModel:
    def __init__(self):
        self.best_threshold = None
        self.name = "LightAutoML"
        self.automl = None
        self.train = None
        self.test = None
        self.features = None
        self.categorical_features = None

    def fit(self):
        if not hasattr(self, 'features') or not self.features:
            self.features = [col for col in self.train.columns if col != 'target']
            print(f"⚠️ Фичи не указаны, используются все колонки кроме target: {self.features}")

        cb = CatBoost()
        lgb = LightGBM()
        cb.params = load_model_params('catboost', MetricNames.f1_1, 0, MetricNames.f1_1)
        lgb.params = load_model_params('lightgbm', MetricNames.f1_1, 0, MetricNames.f1_1)

        cb_params = {
            'default_params': cb.hyper_params
        }
        lgb_params = {
            'default_params': lgb.hyper_params
        }

        # Определяем категориальные признаки

        self.categorical_features = [
            col for col in self.features
            if self.train[col].dtype == 'object' or self.train[col].nunique() < 10
        ]
        # Настройка и обучение LightAutoML
        task = Task('binary')
        roles = {
            'target': 'target',
            'category': self.categorical_features
        }

        train_data = self.train[self.features + ['target']]

        self.automl = TabularAutoML(
            task=task,
            timeout=3600,  # 1 час на обучение
            cpu_limit=4,
            gpu_ids='0',  # Используем GPU
            general_params={"use_algos": [["cb", "lgb"]]},  # Только CatBoost и LightGBM
            lgb_params=lgb_params,
            cb_params=cb_params
        )
        self.automl.fit_predict(train_data, roles=roles)

    def predict_proba(self, X):
        preds = self.automl.predict(X).data
        # Нормализуем вывод к вероятности класса 1
        if preds.ndim == 1:
            return preds.reshape(-1, 1)
        return preds

    def get_score(self):
        preds = self.predict_proba(self.test[self.features])[:, 1]
        return roc_auc_score(self.test['target'], preds)

    def save_model(self, path):
        joblib.dump(self.automl, path)

    def get_feature_importance(self, top_n=15):
        """Получение важности фичей с обработкой ошибок и проверкой столбцов."""
        try:
            feature_scores = self.automl.get_feature_scores()

            # Проверяем наличие столбца с важностью
            importance_col = None
            for col in feature_scores.columns:
                if col.lower() in ["importance", "feature_score", "gain", "weight"]:
                    importance_col = col
                    break

            if importance_col is None:
                print("⚠️ Столбец с важностью фичей не найден. Доступные столбцы:", feature_scores.columns.tolist())
                return pd.DataFrame()

            feature_scores = feature_scores.sort_values(by=importance_col, ascending=False)
            return feature_scores.head(top_n)

        except Exception as e:
            import traceback
            print(f"⚠️ Ошибка при получении важности фичей:\n{traceback.format_exc()}")
            return pd.DataFrame()

    def _plot_metrics(self, y_true, y_proba):
        """Визуализация метрик"""
        plt.figure(figsize=(15, 5))

        # 1. Confusion Matrix
        plt.subplot(131)
        y_pred = (y_proba >= self.best_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix (threshold={self.best_threshold:.2f})")

        # 2. Precision-Recall Curve
        plt.subplot(132)
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')

        # 3. Threshold Analysis
        plt.subplot(133)
        f1_scores = [f1_score(y_true, (y_proba >= t).astype(int)) for t in thresholds[:-1]]
        plt.plot(thresholds[:-1], f1_scores)
        plt.axvline(self.best_threshold, color='r', linestyle='--')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Threshold')

        plt.tight_layout()
        plt.savefig('metrics_plot.png')
        plt.show()

    def get_metrics(self) -> Dict:
        """Расчет метрик с визуализацией"""
        X_test = self.test[self.features]
        y_true = self.test['target']
        y_proba = self.predict_proba(X_test)

        # Если predict_proba возвращает одномерный массив или массив с одной колонкой
        if y_proba.ndim == 1 or y_proba.shape[1] == 1:
            y_proba = y_proba.ravel()  # Преобразуем в одномерный массив
        else:
            y_proba = y_proba[:, 1]  # Берем вероятности класса 1

        # Поиск оптимального порога
        thresholds = np.linspace(0, 1, 100)
        f1_scores = [f1_score(y_true, (y_proba >= t).astype(int)) for t in thresholds]
        self.best_threshold = thresholds[np.argmax(f1_scores)]
        y_pred = (y_proba >= self.best_threshold).astype(int)

        report = classification_report(y_true, y_pred, output_dict=True)

        metrics = {
            'auc': roc_auc_score(y_true, y_proba),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'optimal_threshold': float(self.best_threshold),
            'f1_at_optimal': float(np.max(f1_scores)),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'class_distribution': dict(Counter(y_true)),
            'class_0_metrics': {
                'precision': report['0']['precision'],
                'recall': report['0']['recall'],
                'f1': report['0']['f1-score'],
                'support': report['0']['support']
            },
            'class_1_metrics': {
                'precision': report['1']['precision'],
                'recall': report['1']['recall'],
                'f1': report['1']['f1-score'],
                'support': report['1']['support']
            }
        }

        # Визуализация
        self._plot_metrics(y_true, y_proba)
        return metrics


def load_data(metric=MetricNames.f1_1):
    data_path = Path(__file__).parents[2] / 'data' / 'datasets' / 'hard' / 'sub8'
    features_path = data_path / metric / 'best_features.json'
    with open(features_path, 'r') as f:
        features_data = json.load(f)
    train = pd.read_parquet(os.path.join(data_path, 'cross.parquet'))
    test = pd.read_parquet(os.path.join(data_path, 'test.parquet'))
    return train, test, features_data['set']


def save_results(model, metrics, output_dir):
    """Сохранение результатов с расширенными метриками"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    # Сохраняем модель
    model_path = os.path.join(output_dir, f"model_{timestamp}.joblib")
    model.save_model(model_path)
    print(f'Модель сохранена:{model_path}')

    # Сохраняем метрики
    metrics['timestamp'] = timestamp
    metrics_path = os.path.join(output_dir, f"metrics_{timestamp}.json")

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f'Фичи: {model.features}')
    
    print("\n" + "=" * 70)
    print(f"{'CLASSIFICATION REPORT (LightAutoML)':^70}")
    print(f"{'[Optimal threshold: ' + str(round(metrics['optimal_threshold'], 3)):^70}")
    test_size = sum(len(row) for row in metrics['confusion_matrix'])
    print(f"{f'Test size: {test_size}':^70}")
    print("=" * 70 + "\n")

    # Вывод по классам
    print(f"{'Class 0':<10} | Precision: {metrics['class_0_metrics']['precision']:.4f} | "
          f"Recall: {metrics['class_0_metrics']['recall']:.4f} | "
          f"F1: {metrics['class_0_metrics']['f1']:.4f} | "
          f"Support: {metrics['class_0_metrics']['support']}")

    print(f"{'Class 1':<10} | Precision: {metrics['class_1_metrics']['precision']:.4f} | "
          f"Recall: {metrics['class_1_metrics']['recall']:.4f} | "
          f"F1: {metrics['class_1_metrics']['f1']:.4f} | "
          f"Support: {metrics['class_1_metrics']['support']}\n")


    # Общие метрики
    print(f"{'Overall':<10} | Accuracy: {metrics['accuracy']:.4f} | "
          f"ROC-AUC: {metrics['auc']:.4f} | "
          f"Macro F1: {(metrics['class_0_metrics']['f1'] + metrics['class_1_metrics']['f1']) / 2:.4f}")

    # Confusion Matrix
    print("\n" + "=" * 70)
    print(f"{'CONFUSION MATRIX':^70}")
    print("=" * 70)
    print(f"| {'':<15} | {'Predicted 0':^15} | {'Predicted 1':^15} |")
    print("| " + "-" * 15 + " | " + "-" * 15 + " | " + "-" * 15 + " |")
    print(f"| {'Actual 0':<15} | {metrics['confusion_matrix'][0][0]:^15} | {metrics['confusion_matrix'][0][1]:^15} |")
    print(f"| {'Actual 1':<15} | {metrics['confusion_matrix'][1][0]:^15} | {metrics['confusion_matrix'][1][1]:^15} |")
    print("=" * 70)

    try:
        feat_importance = model.get_feature_importance(top_n=15)
        if not feat_importance.empty:
            print("\n🔝 Топ-15 важных фичей:")
            print(feat_importance.to_markdown(tablefmt="grid"))
        else:
            print("\n⚠️ Информация о важности фичей недоступна")
    except Exception as e:
        print(f"\n⚠️ Ошибка при получении важности фичей: {str(e)}")

    print("\n💾 Результаты сохранены:")
    print(f"- Модель: {model_path}")
    print(f"- Метрики: {metrics_path}")


def learn_light_auto_ml():
    # 1. Загрузка данных
    train, test, features = load_data()

    # 2. Инициализация и обучение модели
    model = LightAutoMLModel()
    model.train = train
    model.test = test
    model.features = features

    print("🚀 Обучение LightAutoML...")
    model.fit()

    metrics = model.get_metrics()

    # 4. Сохранение результатов
    output_dir = os.path.join('data', 'models', 'LightAutoML')
    save_results(model, metrics, output_dir)
