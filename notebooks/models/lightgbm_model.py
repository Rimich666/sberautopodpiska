import joblib
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import cross_val_predict

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

    def convert_categories(self, X):
        """Полностью преобразуем категориальные признаки в числовой формат"""
        X = X.copy()
        initial_size = len(X)
        for col in self._categorical_features:
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
        X_train = self.convert_categories(self.x_train)
        X_test = self.convert_categories(self.x_test)
        self.model.fit(
            X_train,
            self.y_train,
            eval_set=None,
            eval_metric=self.get_eval_metric(self._params.get('metric', MetricNames.f1_1)),
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
        X_train = self.convert_categories(kwargs['X_train'])
        X_val = self.convert_categories(kwargs['X_val'])
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
        X_train_processed = self.convert_categories(X_train)
        X_val_processed = self.convert_categories(X_val)
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

    def save_model(self, path: str) -> None:
        joblib.dump(self.model, path)

    def get_meta(self):
        X_train_processed = self.convert_categories(self.x_train)
        print("Категориальные признаки:", self._categorical_features)
        return cross_val_predict(self.model, X_train_processed, self.y_train, cv=5, method='predict_proba')[:, 1]
