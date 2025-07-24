import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import cross_val_predict
import time
from notebooks.feature_selection import MetricNames
from notebooks.models.model import Model


class CatBoost(Model):
    static_params = {
        'auto_class_weights': 'Balanced',
        'random_seed': 42,
        'task_type': 'GPU',  # Включаем GPU
        'devices': '0',  # Используем первую видеокарту (или '0' для одной GPU)
        'early_stopping_rounds': 50,  # Ранняя остановка для экономии времени
        'border_count': 128,
        'verbose': 0
    }

    def __init__(self):
        super().__init__('catboost')

    @staticmethod
    def get_eval_metric(metric):
        if metric == MetricNames.f1_1:
            return 'F1'
        elif metric == MetricNames.auc:
            return 'AUC'
        elif metric == MetricNames.precision_1:
            return 'Precision'
        elif metric == MetricNames.recall_1:
            return 'Recall'
        return 'F1'

    def convert_categories(self, X):
        return X

    def _set_hyper_params(self):
        self._hyper_params = {
            **CatBoost.static_params,
            'iterations': self._params['iterations'],
            'depth': self._params['depth'],
            'learning_rate': self._params['learning_rate'],
            'l2_leaf_reg': self._params['l2_leaf_reg']
        }
        cat_features_indices = [self._features.index(col) for col in self._categorical_features]
        self._hyper_params.update({'cat_features': cat_features_indices})
        self.model = CatBoostClassifier(**self._hyper_params)

    def fit(self):
        self.model.fit(
            Pool(self.x_train, self.y_train, cat_features=self._categorical_features),
            plot=True
        )

    def set_trial_params(self, trial, **kwargs):
        metric = kwargs.get('metric', MetricNames.f1_1)
        self._trial_params = {
            **CatBoost.static_params,
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
            'iterations': trial.suggest_int('iterations', 300, 1000),
            'eval_metric': Model.get_eval_metric(metric),
            'l2_leaf_reg': trial.suggest_int('l2', 1, 5),
        }

    def fit_trial_model(self, **kwargs):
        model = CatBoostClassifier(**self._trial_params)
        model.fit(kwargs['X_train'], eval_set=kwargs['y_train'])
        return model

    def get_trial_data(self, X_train, y_train, X_val, y_val, cat_features):
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
        return train_pool, X_val, val_pool, y_val

    def set_best_params(self, study, state):
        metric = state.get('metric', MetricNames.f1_1)
        best_params = {
            **CatBoost.static_params,
            'iterations': study.best_params['iterations'],
            'depth': study.best_params['depth'],
            'learning_rate': study.best_params['lr'],
            'l2_leaf_reg': study.best_params['l2'],
            'eval_metric': Model.get_eval_metric(metric),
        }
        self._categorical_features = state['categorical_features']
        self._hyper_params = {
            **best_params
        }

        self.model = CatBoostClassifier(**self._hyper_params)
        return {
            **best_params,
            **state
        }

    def save_model(self, path: str) -> None:
        self.model.save_model(path)

    def get_meta(self):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5)
        meta_preds = np.zeros(len(self.x_train))
        fold_times = []
        print("\n🔹 Начало генерации мета-признаков для CatBoost")
        print(f"🔸 Размер данных: {len(self.x_train)} строк, {len(self.x_train.columns)} признаков")
        print(f"🔸 Категориальные признаки: {self._categorical_features}")
        print(f"🔸 Параметры модели: {self.model.get_params()}")

        for fold_num, (train_idx, val_idx) in enumerate(kf.split(self.x_train), 1):
            fold_start_time = time.time()
            print(f"\n🔹 Обработка fold #{fold_num}")
            print(f"🔸 Размер train: {len(train_idx)}, val: {len(val_idx)}")
            train_pool = Pool(
                self.x_train.iloc[train_idx],
                self.y_train.iloc[train_idx],
                cat_features=self._categorical_features
            )
            val_pool = Pool(
                self.x_train.iloc[val_idx],
                cat_features=self._categorical_features
            )

            model = CatBoostClassifier(**self.model.get_params())
            model.fit(train_pool)
            meta_preds[val_idx] = model.predict_proba(val_pool)[:, 1]
            fold_time = time.time() - fold_start_time
            fold_times.append(fold_time)
            print(f"⏱ Время обработки fold #{fold_num}: {fold_time:.2f} сек")

        total_time = sum(fold_times)
        avg_fold_time = total_time / len(fold_times)
        print(f"\n✅ Все folds обработаны")
        print(f"🔸 Общее время: {total_time:.2f} сек")
        print(f"🔸 Среднее время на fold: {avg_fold_time:.2f} сек")
        print(
            f"🔸 Статистика итоговых мета-признаков: min={meta_preds.min():.4f}, max={meta_preds.max():.4f}, mean={meta_preds.mean():.4f}")

        return meta_preds
