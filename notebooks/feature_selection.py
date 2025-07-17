import json
from pathlib import Path
from typing import Dict
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, classification_report
from notebooks.prepare_sesions import clean_variants, save_dataset, targets


def save_best(best: Dict, path):
    base_path = Path(__file__).parents[1] / 'data' / 'datasets' / path
    save_path = base_path.joinpath('best_features.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(best, f, ensure_ascii=False, indent=4)


def load_dataset(path):
    base_path = Path(__file__).parents[1] / 'data' / 'datasets' / path
    X_TRAIN = pd.read_parquet(Path.joinpath(base_path, 'train.parquet'))
    X_VAL = pd.read_parquet(Path.joinpath(base_path, 'val.parquet'))
    return X_TRAIN, X_VAL


def learn(candidate, train, val):
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

        y_pred = model.predict_proba(X_val)[:, 1]
        report = classification_report(y_val, model.predict(X_val), output_dict=True)
        precision = report['1']['precision']
        auc = roc_auc_score(y_val, y_pred)
        feature['AUC'] = auc
        feature['report'] = report
        print('--------------------------------------------------')
        print(f"{index:2d} ROC-AUC: {auc:.3f}   |   precision: {precision:.5f}   |   Features: {feature_set}")
    print('--------------------------------------------------')
    return candidate


def feature_selection():
    print('версия 2')
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
            columns_list = [{'set': [col], 'AUC': 0} for col in train.columns.to_list() if col not in exceptions]
            best = {'set': [], 'AUC': 0, 'report': {}}
            best_auc = 0
            candidate = columns_list
            while candidate:
                # print(not not candidate)
                # print('Перед учёбой')
                result = sorted(learn(candidate, train, val), key=lambda x: x['AUC'], reverse=True)
                if result[0]:
                    best = result[0]
                    candidate = [{'set': best['set'] + feature['set'][-1:], 'AUC': 0, 'report': {}}
                                 for feature in [x for x in result if (x['AUC'] - best_auc) > 0.001][1:]]
                    for i, c in enumerate(candidate):
                        print(f'{i:2d}', c)
                    best_auc = best['AUC']
                    print(f'Best AUC = {best_auc}')

            print('Best features', best['set'])
            print(f"- ROC-AUC: {best['AUC']:.4f}")
            print("\n📝 Classification Report:")
            df = pd.DataFrame(best['report']).transpose()
            print(df)
            save_best(best, path)


if __name__ == '__main__':
    f = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    for ind, ccc in enumerate(f):
        print(f'{ind:2d}', ccc)
