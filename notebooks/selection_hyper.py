from datetime import datetime
import json
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from matplotlib import pyplot as plt
from optuna.visualization import plot_optimization_history, plot_param_importances
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from notebooks.oversampling import oversample
from notebooks.prepare_sesions import clean_variants, targets
import seaborn as sns

from src.logger import logger

skip = [
    ('lite', 'chat'),
    ('lite', 'sub8'),
    ('hard', 'chat'),
]

MINOR_PARTS = [50]
TRIALS = 20


def load_dataset(path):
    base_path = Path(__file__).parents[1] / 'data' / 'datasets' / path
    X_TRAIN = pd.read_parquet(Path.joinpath(base_path, 'train.parquet'))
    X_VAL = pd.read_parquet(Path.joinpath(base_path, 'val.parquet'))
    X_CROSS = pd.read_parquet(Path.joinpath(base_path, 'cross.parquet'))
    with open(Path.joinpath(base_path, 'best_features.json'), 'r') as f:
        features = json.load(f)
    return X_TRAIN, X_VAL, X_CROSS, features['set']


def create_objective(X_cross, y_cross, cat_features, part):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    for train_idx, val_idx in skf.split(X_cross, y_cross):
        logger.debug(f'Фолдирование. Фолд № {len(folds)}')
        X_train_fold, X_val_fold = X_cross.iloc[train_idx], X_cross.iloc[val_idx]
        y_train_fold, y_val_fold = y_cross.iloc[train_idx], y_cross.iloc[val_idx]
        logger.debug(f'Семплирование. Фолд № {len(folds)}')
        X_train, y_train = oversample(X_train_fold, y_train_fold, cat_features, part)
        logger.debug(f'Пулинг. Фолд № {len(folds)}')
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val_fold, y_val_fold, cat_features=cat_features)
        logger.debug(f'Фолд № {len(folds)} готов к труду и обороне')
        folds.append((train_pool, val_pool, X_val_fold, y_val_fold))

    def objective(trial):
        # 2. Подбираем гиперпараметры
        params = {
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
            'iterations': trial.suggest_int('iterations', 300, 1000),
            'l2_leaf_reg': trial.suggest_int('l2', 1, 5),
            'task_type': 'GPU',  # Включаем GPU
            'devices': '0',  # Используем первую видеокарту (или '0' для одной GPU)
            'early_stopping_rounds': 50,  # Ранняя остановка для экономии времени
            # 'verbose': False,
            'border_count': 128
        }

        # 3. Оценка модели

        scores = []
        for index, (train_pool, val_pool, X_val_fold, y_val_fold) in enumerate(folds):
            logger.debug(f'Кроссвалидация. Фолд № {index}')
            model = CatBoostClassifier(**params, silent=True)
            model.fit(train_pool, eval_set=val_pool, verbose=False)
            scores.append(roc_auc_score(y_val_fold, model.predict_proba(X_val_fold)[:, 1]))
        return np.mean(scores)

    return objective


def run_study(X, y, cat_features, part, trials=50):
    logger.debug(f'Run study. Part = {part}')
    study = optuna.create_study(direction='maximize')
    objective_func = create_objective(X, y, cat_features, part)
    logger.debug(f'Старт оптимизации. Триалов будет {trials}')
    study.optimize(objective_func, n_trials=trials)
    # study.optimize(objective_func, n_trials=2)

    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)

    fig1.show()
    fig2.show()

    return study


def save_result(df, best_params, path):
    results_dir = Path(__file__).parents[1] / 'data' / 'models' / 'hyper' / path
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


def hyper_select():
    logger.debug('Старт подбора. Версия 7')
    for variant in clean_variants:
        for target in targets:
            if (variant.name, target.name) in skip:
                print((variant.name, target.name))
                continue
            path = Path(variant.name).joinpath(target.name)
            try:
                train, val, cross, features_set = load_dataset(path)
                logger.info("✅ Данные загружены!")
                print(f"Train: {train.shape[0]} строк, Validation: {val.shape[0]} строк, Cross: {cross.shape[0]} строк")
            except Exception as e:
                print(f"❌ Ошибка: {e}")
            logger.debug(f'MINOR_PARTS = {MINOR_PARTS}')
            for part in MINOR_PARTS:
                save_path = path.joinpath(str(part))
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                y_train = train['target']
                y_test = val['target']
                y_cross = cross['target']
                X_train = train[features_set]
                X_test = val[features_set]
                X_cross = cross[features_set]
                categorical_indexes = [i for i, col in enumerate(X_cross.columns)
                                       if X_cross[col].dtype == 'object' or X_cross[col].nunique() < 10]
                cat_features = [col for i, col in enumerate(X_cross.columns) if i in categorical_indexes]

                features = X_cross.columns.tolist()
                study = run_study(X_cross, y_cross, categorical_indexes, part, TRIALS)
                results_df = study.trials_dataframe()

                print("Все параметры из study.best_params:", list(study.best_params.keys()))

                best_params = {
                    'depth': study.best_params['depth'],
                    'iterations': study.best_params['iterations'],
                    'l2_leaf_reg': study.best_params['l2'],
                    'learning_rate': study.best_params['lr'],
                    'roc_auc': study.best_value,
                    'timestamp': timestamp,
                    'features_used': features,
                    'categorical_features': cat_features
                }

                save_result(results_df, best_params, save_path)

                print("\nЛучшие параметры:")
                for key, value in study.best_params.items():
                    print(f"{key}: {value}")

                best_trial = study.best_trial

                print("\nЗначимость фичей из лучшей модели:")
                best_model = CatBoostClassifier(
                    **{k: v for k, v in best_trial.params.items()
                       if k in ['depth', 'learning_rate', 'iterations', 'l2_leaf_reg']},
                    cat_features=cat_features,
                    verbose=False
                )

                X, y = oversample(X_train, y_train, cat_features, part)

                best_model.fit(X, y)
                y_pred = best_model.predict(X_test)
                y_proba = best_model.predict_proba(X_test)[:, 1]

                feature_importances = pd.DataFrame({
                    'feature': features,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)

                print(feature_importances.to_string(index=False))
                print("\n🔍 Подробные метрики лучшей модели:")
                print("=" * 60)
                print("📊 Classification Report:")
                print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))

                print("\n📈 ROC-AUC Score:", roc_auc_score(y_test, y_proba))

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Predicted 0', 'Predicted 1'],
                            yticklabels=['Actual 0', 'Actual 1'])
                plt.title("Confusion Matrix")
                plt.show()
