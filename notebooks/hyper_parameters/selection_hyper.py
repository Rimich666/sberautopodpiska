from datetime import datetime
import json
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
from matplotlib import pyplot as plt
from optuna.visualization import plot_optimization_history, plot_param_importances
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from notebooks.feature_selection import MetricNames
from notebooks.models.model import Model
from notebooks.models.models import models
from notebooks.oversampling import oversample
from notebooks.prepare_sesions import clean_variants, targets
import seaborn as sns
from src.logger import logger

skip = [
    ('lite', 'chat'),
    ('lite', 'sub8'),
    ('hard', 'chat'),
]

MINOR_PARTS = (0, 5, 10, 25, 50)
TRIALS = 20
VALID_METRICS = [MetricNames.auc, MetricNames.precision_1, MetricNames.recall_1, MetricNames.f1_1]
DEFAULT_METRIC = MetricNames.auc


def load_dataset(path, features_metric):
    base_path = Path(__file__).parents[2] / 'data' / 'datasets' / path
    features_path = base_path if not features_metric else (base_path.joinpath(features_metric))
    X_TRAIN = pd.read_parquet(Path.joinpath(base_path, 'train.parquet'))
    X_VAL = pd.read_parquet(Path.joinpath(base_path, 'val.parquet'))
    X_CROSS = pd.read_parquet(Path.joinpath(base_path, 'cross.parquet'))
    with open(Path.joinpath(features_path, 'best_features.json'), 'r') as f:
        features = json.load(f)
    return X_TRAIN, X_VAL, X_CROSS, features['set']


def create_objective(model_instance: Model, cat_features, part, metric):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    if metric not in VALID_METRICS:
        raise ValueError(f"Неподдерживаемая метрика: {metric}. Допустимые значения: {VALID_METRICS}")

    for train_idx, val_idx in skf.split(model_instance.x_cross, model_instance.y_cross):
        X_train_fold, X_val_fold = model_instance.x_cross.iloc[train_idx], model_instance.x_cross.iloc[val_idx]
        y_train_fold, y_val_fold = model_instance.y_cross.iloc[train_idx], model_instance.y_cross.iloc[val_idx]
        X_train_oversampled, y_train_oversampled = oversample(X_train_fold, y_train_fold, cat_features, part)

        X_train_processed, X_val_processed, y_train_processed, y_val_processed = model_instance.get_trial_data(
            X_train_oversampled, y_train_oversampled, X_val_fold, y_val_fold, cat_features)

        logger.debug(f'Фолд № {len(folds)} готов к труду и обороне')
        folds.append((X_train_processed, y_train_processed, X_val_processed, y_val_processed))

    def objective(trial):
        # 2. Подбираем гиперпараметры
        model_instance.set_trial_params(trial=trial, metric=metric)

        # 3. Оценка модели

        scores = []
        for index, (X_train, y_train, X_val, y_val) in enumerate(folds):
            # logger.debug(f'Кроссвалидация. Фолд № {index}')
            model = model_instance.fit_trial_model(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

            assert len(X_val) == len(y_val), f"Несоответствие размеров: X_val={len(X_val)}, y_val={len(y_val)}"

            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]

            if len(y_val) != len(y_pred):
                raise ValueError(
                    f"Несовпадение размеров при предсказании: "
                    f"y_val={len(y_val)}, y_pred={len(y_pred)}\n"
                    f"Индексы y_val: {y_val.index[:5]}, X_val: {X_val.index[:5]}"
                )

            if metric == MetricNames.auc:
                score = roc_auc_score(y_val, y_proba)
            elif metric == MetricNames.precision_1:
                score = precision_score(y_val, y_pred, pos_label=1)
            elif metric == MetricNames.recall_1:
                score = recall_score(y_val, y_pred, pos_label=1)
            elif metric == MetricNames.f1_1:
                score = f1_score(y_val, y_pred, pos_label=1)

            scores.append(score)
        return np.mean(scores)

    return objective


def run_study(model, cat_features, part, metric, trials=50):
    logger.debug(f'Run study для модели {model.name}. Part = {part}')
    study = optuna.create_study(direction='maximize')
    objective_func = create_objective(model, cat_features, part, metric)
    logger.debug(f'Старт оптимизации. Триалов будет {trials}')
    study.optimize(objective_func, n_trials=trials)
    # study.optimize(objective_func, n_trials=2)

    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)

    fig1.show()
    fig2.show()

    return study


def save_result(df, best_params, path):
    results_dir = Path(__file__).parents[2] / 'data' / 'models' / 'hyper' / path
    Path.mkdir(results_dir, parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"best_params_{timestamp}.json"
    csv_filename = f"optimization_results_{timestamp}.csv"
    json_path = results_dir.joinpath(json_filename)
    csv_path = results_dir.joinpath(csv_filename)
    df.to_csv(csv_path, index=False)
    with open(json_path, 'w') as f:
        json.dump(best_params, f, indent=4)

    print(f"\nРезультаты сохранены в:")
    print(f"- Параметры модели: {json_path}")
    print(f"- Полные результаты: {csv_path}")


def hyper_select(
        model: Model = models.catboost,
        metric: MetricNames = DEFAULT_METRIC,
        trial_count=TRIALS,
        features_metric: MetricNames = None,
        parts=MINOR_PARTS):
    """
        Принимает модель,
        Метрику для оптимизации,
        количество триалов,
        метрику отбора фичей, для подгрузки этих самых фмчей
        долю, которую хочет занять минорный класс в датасете
    """
    logger.debug(f'Старт подбора. Версия 8.01. Метрика оптимизации: {metric}')
    for variant in clean_variants:
        for target in targets:
            if (variant.name, target.name) in skip:
                print((variant.name, target.name))
                continue
            path = Path(variant.name).joinpath(target.name)
            try:
                train, test, cross, model.features = load_dataset(path, features_metric)
                logger.info("✅ Данные загружены!")
                model.train = train
                model.test = test
                model.cross = cross
                print(f"Train: {model.train.shape[0]} строк, Validation: {model.test.shape[0]} строк,"
                      f" Cross: {model.cross.shape[0]} строк")
            except Exception as e:
                print(f"❌ Ошибка: {e}")
            logger.debug(f'MINOR_PARTS = {parts}')
            for part in parts:
                save_path = path if not features_metric else path.joinpath(features_metric)
                save_path = Path(model.name).joinpath(save_path).joinpath(str(part), metric)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                categorical_indexes = [i for i, col in enumerate(model.x_cross.columns)
                                       if model.x_cross[col].dtype == 'object' or model.x_cross[col].nunique() < 10]
                cat_features = [col for i, col in enumerate(model.x_cross.columns) if i in categorical_indexes]

                study = run_study(model, categorical_indexes, part, metric, trial_count)
                results_df = study.trials_dataframe()

                features = model.x_cross.columns.tolist()
                state = {
                    metric: study.best_value,
                    'timestamp': timestamp,
                    'trials_count': trial_count,
                    'features_used': features,
                    'categorical_features': cat_features,
                    'metric': metric
                }
                best_params = model.set_best_params(study, state)

                model.x_train, model.y_train = oversample(model.x_train, model.y_train, cat_features, part)

                model.fit()

                print("Model:", model.model)
                print("Все параметры из study.best_params:", list(study.best_params.keys()))

                feature_importances = pd.DataFrame({
                    'feature': features,
                    'importance': model.model.feature_importances_
                }).sort_values('importance', ascending=False)
                print(f'\n🔍 Лучшие параметры для target: {target.name}, variant: {variant.name}, parts count: {part}')
                if features_metric:
                    print(f'Метрика отбора фичей: {features_metric}')
                print(f'Метрика оптимизации: {metric}')
                for key, value in study.best_params.items():
                    print(f'- {key}: {value}')

                report, cm = model.get_optimal_report()
                best_params['optimal_threshold'] = model.optimal_threshold
                save_result(results_df, best_params, save_path)
                print(f'Оптимальный порог: {model.optimal_threshold}')
                print("\n📊 Значимость фичей из лучшей модели:")
                print(feature_importances.to_string(index=False))
                print("\n🔍 Подробные метрики лучшей модели:")
                print("=" * 60)
                print("📊 Classification Report:")
                print(model.get_report())
                print(f"\n📝 Classification Report с оптимизированным порогом: {model.optimal_threshold}:")
                print(report)
                print("\n📈 ROC-AUC Score:", model.get_score(MetricNames.auc))

                # Confusion Matrix
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Predicted 0', 'Predicted 1'],
                            yticklabels=['Actual 0', 'Actual 1'])
                plt.title("Confusion Matrix")
                plt.show()
