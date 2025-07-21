from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import optuna
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def create_lgbm_objective(X_cross, y_cross, cat_features, part, metric):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    
    for train_idx, val_idx in skf.split(X_cross, y_cross):
        X_train_fold, X_val_fold = X_cross.iloc[train_idx], X_cross.iloc[val_idx]
        y_train_fold, y_val_fold = y_cross.iloc[train_idx], y_cross.iloc[val_idx]
        
        # Применяем oversampling (если нужно)
        X_train, y_train = oversample(X_train_fold, y_train_fold, cat_features, part)
        
        # Для LightGBM преобразуем категориальные признаки
        for col in cat_features:
            X_train[col] = X_train[col].astype('category')
            X_val_fold[col] = X_val_fold[col].astype('category')
        
        folds.append((X_train, X_val_fold, y_train, y_val_fold))

    def objective(trial):
        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'metric': 'binary_logloss',
            'class_weight': 'balanced',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'device_type': 'gpu',  # Используем GPU
            'random_state': 42,
            'verbosity': -1
        }

        scores = []
        for X_train, X_val, y_train, y_val in folds:
            model = LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )

            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]

            if metric == 'roc_auc':
                score = roc_auc_score(y_val, y_proba)
            elif metric == 'precision':
                score = precision_score(y_val, y_pred, pos_label=1)
            elif metric == 'recall':
                score = recall_score(y_val, y_pred, pos_label=1)
            elif metric == 'f1':
                score = f1_score(y_val, y_pred, pos_label=1)

            scores.append(score)
        
        return np.mean(scores)

    return objective

def run_lgbm_study(X, y, cat_features, part, metric, trials=50):
    logger.debug(f'Run LGBM study. Part = {part}')
    study = optuna.create_study(direction='maximize')
    objective_func = create_lgbm_objective(X, y, cat_features, part, metric)
    study.optimize(objective_func, n_trials=trials)
    
    # Визуализация
    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)
    fig1.show()
    fig2.show()
    
    return study

def lgbm_hyper_select(
        metric=DEFAULT_METRIC,
        trial_count=TRIALS,
        features_metric: MetricNames = None,
        parts=MINOR_PARTS):
    
    logger.debug(f'Старт подбора LGBM. Метрика оптимизации: {metric}')
    
    for variant in clean_variants:
        for target in targets:
            if (variant.name, target.name) in skip:
                continue
                
            path = Path(variant.name).joinpath(target.name)
            try:
                train, val, cross, features_set = load_dataset(path, features_metric)
                logger.info("✅ Данные загружены!")
            except Exception as e:
                print(f"❌ Ошибка: {e}")
                continue

            for part in parts:
                save_path = path if not features_metric else path.joinpath(features_metric)
                save_path = save_path.joinpath(str(part), metric, 'lgbm')
                
                y_train = train['target']
                y_test = val['target']
                y_cross = cross['target']
                X_train = train[features_set]
                X_test = val[features_set]
                X_cross = cross[features_set]
                
                categorical_indexes = [i for i, col in enumerate(X_cross.columns) 
                                     if X_cross[col].dtype == 'object' or X_cross[col].nunique() < 10]
                cat_features = [col for i, col in enumerate(X_cross.columns) if i in categorical_indexes]

                # Запуск оптимизации
                study = run_lgbm_study(X_cross, y_cross, categorical_indexes, part, metric, trial_count)
                results_df = study.trials_dataframe()

                # Сохранение результатов
                best_params = {
                    'objective': 'binary',
                    'boosting_type': 'gbdt',
                    'metric': 'binary_logloss',
                    'n_estimators': study.best_params['n_estimators'],
                    'learning_rate': study.best_params['learning_rate'],
                    'num_leaves': study.best_params['num_leaves'],
                    'max_depth': study.best_params['max_depth'],
                    'min_child_samples': study.best_params['min_child_samples'],
                    'reg_alpha': study.best_params['reg_alpha'],
                    'reg_lambda': study.best_params['reg_lambda'],
                    'subsample': study.best_params['subsample'],
                    'colsample_bytree': study.best_params['colsample_bytree'],
                    'device_type': 'gpu',
                    'random_state': 42,
                    'class_weight': 'balanced'
                }

                save_result(results_df, best_params, save_path)

                # Обучение и оценка лучшей модели
                best_model = LGBMClassifier(**best_params)
                X, y = oversample(X_train, y_train, cat_features, part)
                
                # Преобразование категориальных признаков
                for col in cat_features:
                    X[col] = X[col].astype('category')
                    X_test[col] = X_test[col].astype('category')
                
                best_model.fit(X, y)

                y_pred = best_model.predict(X_test)
                y_proba = best_model.predict_proba(X_test)[:, 1]

                # Вывод результатов
                print(f"\n🔍 LGBM Результаты для target: {target.name}, variant: {variant.name}, part: {part}")
                print(classification_report(y_test, y_pred))
                print("ROC-AUC:", roc_auc_score(y_test, y_proba))
                
                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title("LGBM Confusion Matrix")
                plt.show()