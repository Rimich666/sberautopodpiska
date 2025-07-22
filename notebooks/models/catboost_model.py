from catboost import CatBoostClassifier, Pool
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

    def _convert_categories(self, X, cat_features=None):
        return X

    def _set_hyper_params(self):
        self._hyper_params = {
            **CatBoost.static_params,
            'iterations': self._params['iterations'],
            'depth': self._params['depth'],
            'learning_rate': self._params['learning_rate'],
            'l2_leaf_reg': self._params['l2_leaf_reg']
        }
        self.model = CatBoostClassifier(**self._hyper_params)

    def fit(self):
        if self.name == 'catboost':
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
