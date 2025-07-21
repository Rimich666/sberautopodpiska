import json
from pathlib import Path
from typing import Dict
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score, precision_score
from notebooks.prepare_sesions import clean_variants, save_dataset, targets


class MetricNames:
    precision_1 = 'precision_1'
    recall_1 = 'recall_1'
    f1_1 = 'f1_1'
    precision_0 = 'precision_0'
    recall_0 = 'recall_0'
    f1_0 = 'f1_0'
    accuracy = 'accuracy'
    auc = 'auc'
    pr_auc = 'pr_auc'


class Metrics:

    def __init__(self, report, auc, pr_auc):
        self.precision_1 = report['1']['precision']
        self.recall_1 = report['1']['recall']
        self.f1_1 = report['1']['f1-score']
        self.precision_0 = report['0']['precision']
        self.recall_0 = report['0']['recall']
        self.f1_0 = report['0']['f1-score']
        self.accuracy = report['accuracy']
        self.auc = auc
        self.pr_auc = pr_auc

    @property
    def dict(self) -> dict:
        return {
            'AUC': self.auc,
            'PR_AUC': self.pr_auc,
            'report': {
                '0': {
                    'precision': self.precision_0,
                    'recall': self.recall_0,
                    'f1-score': self.f1_0,
                },
                '1': {
                    'precision': self.precision_1,
                    'recall': self.recall_1,
                    'f1-score': self.f1_1,
                }
            },
            'accuracy': self.accuracy,
        }

    def get(self, metric_name: str) -> float:
        return self.__getattribute__(metric_name)


def save_best(best: Dict, path):
    base_path = Path(__file__).parents[1] / 'data' / 'datasets' / path
    Path.mkdir(base_path, parents=True, exist_ok=True)
    save_path = base_path.joinpath('best_features.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(best, f, ensure_ascii=False, indent=4)


def load_dataset(path):
    base_path = Path(__file__).parents[1] / 'data' / 'datasets' / path
    x_train = pd.read_parquet(Path.joinpath(base_path, 'train.parquet'))
    x_val = pd.read_parquet(Path.joinpath(base_path, 'val.parquet'))
    return x_train, x_val


def calculate_metrics(y_true, y_pred, y_proba) -> Metrics:
    """Calculate multiple metrics for model evaluation."""
    report = classification_report(y_true, y_pred, output_dict=True)
    auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    metrics = Metrics(report, auc, pr_auc)

    return metrics


def learn(candidate, train, val, target_metric=MetricNames.auc):
    for index, feature in enumerate(candidate):
        feature_set = feature['set']
        # print(f"Feature set {feature_set}")
        X_train = train[feature_set]
        y_train = train['target']
        y_val = val['target']
        X_val = val[feature_set]
        model = CatBoostClassifier(
            cat_features=[f for f in feature_set if X_train[f].dtype == 'object'],
            # scale_pos_weight=scale_pos_weight,
            auto_class_weights='Balanced',
            task_type='GPU',
            devices='0',
            verbose=0
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        # y_pred = model.predict_proba(X_val)[:, 1]
        metrics = calculate_metrics(y_val, y_pred, y_proba)

        feature.update({
            'metrics': metrics,
            'score': metrics.get(target_metric)
        })
        print('-' * 80)
        print(f"{index:2d} Features: {feature_set}")
        print(f"f1: {metrics.get(MetricNames.f1_1):.4f} precision: {metrics.get(MetricNames.precision_1):.4f}"
              f" recall: {metrics.get(MetricNames.recall_1):.4f} roc-auc: {metrics.get(MetricNames.auc):.4f}"
              f" pr-auc: {metrics.get(MetricNames.pr_auc):.4f}")
    print('-' * 80)
    return candidate


def feature_selection(target_metric: MetricNames.auc, min_improvement: float = 0.001):
    print('версия 5.12 (с PR-AUC)')
    exceptions = ['client_id', 'session_id', 'target', 'visit_date', 'visit_time']
    for variant in clean_variants:
        for target in targets:
            path = Path(variant.name).joinpath(target.name)
            try:
                train, val = load_dataset(path)
                print("✅ Данные загружены!")
                print(f"Train: {train.shape[0]} строк, Validation: {val.shape[0]} строк")
            except Exception as e:
                print(f"❌ Ошибка: {e}")
            columns_list = [{'set': [col], 'score': 0} for col in train.columns.to_list() if col not in exceptions]
            best = {'set': [], 'score': 0, 'metrics': None}
            best_score = 0
            candidate = columns_list
            while candidate:
                result = sorted(learn(candidate, train, val, target_metric), key=lambda x: x['score'], reverse=True)
                if result:
                    best = result[0]
                    candidate = [{'set': best['set'] + feature['set'][-1:], 'score': 0, 'metrics': None}
                                 for feature in [x for x in result if (x['score'] - best_score) > min_improvement][1:]]
                    for i, c in enumerate(candidate):
                        print(f'{i:2d}', c)
                    best_score = best['score']
                    print(f'Best {target_metric} = {best_score}')

            print(f'\n🔍 Параметры модели для target: {target.name}, variant: {variant.name}')
            print("=" * 60)
            print(f'\n📈  Best features: {best["set"]}')
            print(f'📈  Best {target_metric.upper()}: {best['score']}')
            print(f'📈  ROC-AUC: {best['metrics'].auc}')
            print(f'📈  PR-AUC: {best['metrics'].pr_auc}')
            df = pd.DataFrame(best['metrics'].dict['report']).transpose()
            print("\n📊 Отчёт по метрикам:")
            print("=" * 60)
            print(df)
            best.update({
                'metrics': best['metrics'].dict
            })
            save_best(best, path.joinpath(target_metric))


if __name__ == '__main__':
    f = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    for ind, ccc in enumerate(f):
        print(f'{ind:2d}', ccc)
