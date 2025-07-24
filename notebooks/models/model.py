from abc import ABC, abstractmethod
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from sklearn.model_selection import cross_val_predict

from notebooks.feature_selection import MetricNames
from notebooks.models.helpers import find_optimal_threshold


class Model(ABC):
    def __init__(self, name):
        self.y_cross = None
        self.x_cross = None
        self._folds = None
        self._trial_params = None
        self._hyper_params = None
        self._cross = None
        self._optimal_threshold = None
        self.model = None
        self.name = name
        self._train = None
        self.x_train = None
        self.y_train = None
        self._test = None
        self.x_test = None
        self.y_test = None
        self._params = {}
        self._features = []
        self._categorical_features = []

    @staticmethod
    @abstractmethod
    def get_eval_metric(metric):
        pass

    @abstractmethod
    def convert_categories(self, X):
        pass

    @abstractmethod
    def _set_hyper_params(self):
        pass

    @abstractmethod
    def set_trial_params(self, trial, **kwargs):
        pass

    @abstractmethod
    def get_trial_data(self, X_train, X_val, y_train, y_val, cat_features):
        pass

    @abstractmethod
    def set_best_params(self, study, state):
        pass

    @abstractmethod
    def fit(self):
        pass

    @property
    def cross(self):
        return self._cross

    @cross.setter
    def cross(self, value):
        self._cross = value
        self.x_cross = value[self._features]
        self.y_cross = value['target']

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, value):
        self._train = value
        self.x_train = value[self._features]
        self.y_train = value['target']

    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, value):
        self._test = value
        self.x_test = value[self._features]
        self.y_test = value['target']

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = value
        self._features = self._params['features_used']
        self._categorical_features = self._params['categorical_features']
        self._set_hyper_params()
        self._optimal_threshold = self._params['optimal_threshold']

    @property
    def hyper_params(self):
        return self._hyper_params

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self._features = value

    @property
    def categorical_features(self):
        return self._categorical_features

    @property
    def optimal_threshold(self):
        return self._optimal_threshold

    def get_score(self, metric_name: MetricNames) -> float:
        x_test_converted = self.convert_categories(self.x_test)
        if metric_name == MetricNames.auc:
            return roc_auc_score(self.y_test, self.model.predict_proba(x_test_converted)[:, 1])
        y_pred = self.model.predict(x_test_converted)
        if metric_name == MetricNames.precision_1:
            return precision_score(self.y_test, y_pred, pos_label=1)
        if metric_name == MetricNames.recall_1:
            return recall_score(self.y_test, y_pred, pos_label=1)
        if metric_name == MetricNames.f1_1:
            return f1_score(self.y_test, y_pred, pos_label=1)

    @abstractmethod
    def fit_trial_model(self):
        pass

    def get_optimal_report(self, output_dict=False):
        x_test_converted = self.convert_categories(self.x_test)
        y_proba = self.model.predict_proba(x_test_converted)[:, 1]
        if not self._optimal_threshold:
            self._optimal_threshold, _ = find_optimal_threshold(self.y_test, y_proba)
        y_pred_optimal = (y_proba >= self._optimal_threshold).astype(int)
        return classification_report(
            self.y_test,
            y_pred_optimal,
            target_names=['Class 0', 'Class 1'],
            output_dict=output_dict
        ), confusion_matrix(self.y_test, y_pred_optimal)

    def get_report(self, output_dict=False):
        x_test_converted = self.convert_categories(self.x_test)
        return classification_report(
            self.y_test,
            self.model.predict(x_test_converted),
            target_names=['Class 0', 'Class 1'],
            output_dict=output_dict
        )

    @abstractmethod
    def save_model(self, path: str) -> None:
        pass

    def get_meta(self):
        X_train_processed = self.convert_categories(self.x_train)
        return cross_val_predict(
            self.model, X_train_processed, self.y_train, cv=5, method='predict_proba', n_jobs=-1
        )[:, 1]
