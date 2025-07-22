import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from notebooks.feature_selection import MetricNames
from notebooks.models.model import Model


class LightGBM(Model):
    static_params = {
        'objective': 'binary',
        'random_state': 42,
        'device': 'gpu',  # Используем GPU
        'n_jobs': -1,     # Все ядра CPU
        'verbosity': -1,   # Без логов
        'boosting_type': 'gbdt',
        'class_weight': 'balanced'
    }

    def __init__(self):
        super().__init__('lightgbm')
        self._category_mapping = {}

    @staticmethod
    def get_eval_metric(metric):
        if metric == MetricNames.f1_1:
            return 'f1'
        elif metric == MetricNames.auc:
            return 'auc'
        elif metric == MetricNames.precision_1:
            return 'average_precision'  # ближайший аналог precision
        elif metric == MetricNames.recall_1:
            return 'recall'
        return 'binary_logloss'  # по умолчанию

    def _convert_categories(self, X, cat_features):
        """Полностью преобразуем категориальные признаки в числовой формат"""
        X = X.copy()
        initial_size = len(X)
        for col in cat_features:
            if col in X.columns:
                if X[col].dtype == 'object' or X[col].nunique() < 100:
                    if col not in self._category_mapping:
                        self._category_mapping[col] = {v: i for i, v in enumerate(X[col].astype(str).unique())}
                    X[col] = X[col].astype(str).map(self._category_mapping[col]).fillna(-1).astype(int)
                elif X[col].dtype == 'bool':
                    X[col] = X[col].astype(int)

        if len(X) != initial_size:
            raise ValueError(f"Размер данных изменился с {initial_size} на {len(X)} после преобразования категорий")

        return X

    def _set_hyper_params(self):
        self._hyper_params = {
            **LightGBM.static_params,
            'n_estimators': self._params['n_estimators'],
            'max_depth': self._params['max_depth'],
            'learning_rate': self._params['learning_rate'],
            'reg_lambda': self._params['reg_lambda']
        }
        self.model = LGBMClassifier(**self._hyper_params)

    def fit(self):
        X_train = self._convert_categories(self.x_train, self._categorical_features)
        X_test = self._convert_categories(self.x_test, self._categorical_features)
        print("Типы данных перед обучением:")
        print(X_train.dtypes)

        print("\nТипы данных в тесте:")
        print(X_test.dtypes)
        # print("Уникальные категории в X_test:", X_test.nunique())
        # print(self.x_test.shape)
        # print(self.get_eval_metric(self._params.get('metric', MetricNames.f1_1)))
        # print(self.y_test.shape)
        # print(type(X_test))
        # print(type(self.y_test))
        # print("NaN в X_test:", X_test.isna().sum().sum())
        # print("Inf в X_test:", np.isinf(X_test.values).sum())
        # print("Params in fit():", self.model.get_params())
        # print("Params in fit_trial_model:", self._trial_params)
        self.model.fit(
            X_train,
            self.y_train,
            eval_set=None,
            # eval_set=[(X_test, self.y_test)],
            eval_metric=self.get_eval_metric(self._params.get('metric', MetricNames.f1_1)),
            # callbacks=[
            #     early_stopping(stopping_rounds=50, verbose=False)
            # ]
            callbacks=None
        )

    def set_trial_params(self, trial, **kwargs):
        metric = kwargs.get('metric', MetricNames.f1_1)
        self._trial_params = {
            **LightGBM.static_params,
            'max_depth': trial.suggest_int('max_depth', 4, 8),  # Единый стиль
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
            'reg_lambda': trial.suggest_int('reg_lambda', 1, 5),
            'metric': self.get_eval_metric(metric)
        }

    def fit_trial_model(self, **kwargs):
        model = LGBMClassifier(**self._trial_params)
        X_train = self._convert_categories(kwargs['X_train'], self._categorical_features)
        X_val = self._convert_categories(kwargs['X_val'], self._categorical_features)
        model.fit(
            X_train,
            kwargs['y_train'],
            eval_set=[(X_val, kwargs['y_val'])],
            eval_metric=self.get_eval_metric(self._trial_params.get('metric', MetricNames.f1_1)),
            callbacks=[
                early_stopping(stopping_rounds=50, verbose=False)
            ]
        )
        return model

    def get_trial_data(self, X_train, y_train, X_val, y_val, cat_features):
        self._categorical_features = [X_train.columns[i] for i in cat_features if i < len(X_train.columns)]
        X_train_processed = self._convert_categories(X_train, self._categorical_features)
        X_val_processed = self._convert_categories(X_val, self._categorical_features)
        return X_train_processed, X_val_processed, y_train, y_val

    def set_best_params(self, study, state):
        metric = state.get('metric', MetricNames.f1_1)
        best_params = {
            **LightGBM.static_params,
            'max_depth': study.best_params['max_depth'],
            'learning_rate': study.best_params['learning_rate'],
            'n_estimators': study.best_params['n_estimators'],
            'reg_lambda': study.best_params['reg_lambda'],
            'metric': self.get_eval_metric(state.get('metric', MetricNames.f1_1))
        }
        self._categorical_features = state['categorical_features']
        self._hyper_params = best_params
        self.model = LGBMClassifier(**self._hyper_params)
        return {**best_params, **state}
