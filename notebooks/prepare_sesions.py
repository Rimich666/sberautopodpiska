from pathlib import Path
from typing import Callable
from sklearn.model_selection import train_test_split
from notebooks.helpers import get_season, is_peak_hour, time_of_day


class Target:
    def __init__(self, name, columns):
        self.name: str = name
        self.columns: [str] = columns


class CleanVariant:
    def __init__(self, name, executor):
        self.name: str = name
        self.exec: Callable = executor


targets: [Target] = [
    Target('chat', ['start_chat']),
    Target('sub8', [
        'sub_car_claim_click',
        'sub_car_claim_submit_click',
        'sub_open_dialog_click',
        'sub_custom_question_submit_click',
        'sub_call_number_click',
        'sub_callback_submit_click',
        'sub_submit_success',
        'sub_car_request_submit_click'
    ])
]


def lite_clean(df):
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna('unknown')
    return df


def hard_clean(df):
    df['device_model'] = df['device_model'].fillna('noname')
    df.loc[df['device_os'].isnull() & (df['device_brand'] == 'Apple'), 'device_os'] = 'iOS'
    mask = df['device_brand'].isnull() & df['device_os'].isnull()
    df.loc[mask, ['device_brand', 'device_os']] = ['noname_device_brand', 'noname_device_os']
    desktop_oses = ['Windows', 'Macintosh', 'Linux', 'Chrome OS']
    df.loc[df['device_os'].isin(desktop_oses), 'device_brand'] = 'PC'
    mask = (df['device_os'].isnull() & df['device_brand'].notnull() & (df['device_brand'] != '(not set)'))
    df.loc[mask, 'device_os'] = 'Android'
    df = df[df['device_brand'].notnull()]
    df = df.loc[~(df['device_brand'] == '(not set)')].reset_index(drop=True)
    df['utm_keyword'] = df['utm_keyword'].fillna('without_utm_keyword')
    df = df[df['utm_source'].notnull()]
    df.loc[df['utm_campaign'].isnull(), 'utm_campaign'] = 'without_utm_campaign'

    # Заполняем их значением 'without_utm_campaign'
    df.loc[df['utm_adcontent'].isnull(), 'utm_adcontent'] = 'without_utm_adcontent'
    return df


clean_variants: [CleanVariant] = [
    CleanVariant('lite', lite_clean),
    CleanVariant('hard', hard_clean)
]


def add_features(df):
    df['is_returning'] = (df['visit_number'] > 1).astype(int)
    df['brand_tier'] = df['device_brand'].map({
        'Apple': 'premium',
        'Samsung': 'premium',
        'Huawei': 'mid',
        'Xiaomi': 'mid'
    }).fillna('other')
    df['frequent_visitor'] = (df['visit_number'] >= 3).astype(int)
    df['visit_month'] = df['visit_date'].apply(lambda x: x.month)
    df['visit_season'] = df['visit_month'].apply(get_season)
    df['visit_day_week'] = df['visit_date'].apply(lambda x: x.weekday())
    df['is_weekend'] = df['visit_day_week'].isin([5, 6]).astype(int)
    df['visit_hour'] = df['visit_time'].apply(lambda x: x.hour)
    df['is_peak_hour'] = df['visit_hour'].apply(is_peak_hour)
    df['time_of_day'] = df['visit_hour'].apply(time_of_day)
    df = df.drop(['visit_hour'], axis=1)
    df['has_utm_keyword'] = df['utm_keyword'].notna().astype(int)
    df['has_utm_campaign'] = df['utm_campaign'].notna().astype(int)
    return df


def save_dataset(df, path, name):
    base_path = Path(__file__).parents[1] / 'data' / 'datasets' / path
    Path.mkdir(base_path, parents=True, exist_ok=True)
    save_path = Path.joinpath(base_path, name + '.parquet')
    df.to_parquet(save_path, index=False)
    print(f'{name} сохранён в {save_path}')


def split_ds(df):
    TEST = 0.2
    X_CROSS, X_TEST, y_temp, Y_test = train_test_split(
        df,
        df['target'],
        test_size=TEST,
        stratify=df['target'],
        random_state=42
    )
    X_TRAIN, X_VAL, Y_train, Y_val = train_test_split(
        X_CROSS,
        y_temp,
        test_size=TEST / (1 - TEST),
        stratify=y_temp,  # Стратификация снова
        random_state=42
    )
    return X_TRAIN, X_VAL, X_TEST, X_CROSS


def prepare_sessions(sessions, hits):
    valid_session_ids = hits['session_id'].unique()
    df = sessions[sessions['session_id'].isin(valid_session_ids)].copy()
    df = add_features(df)
    for variant in clean_variants:
        df_clean = variant.exec(df)
        save_dataset(df_clean, variant.name, 'full')
        for target in targets:
            target_sessions = hits[hits['event_action'].isin(target.columns)]['session_id'].unique()
            df_clean['target'] = df_clean['session_id'].isin(target_sessions).astype(int)
            print(f'Доля сессий с {target.name} и вариантом очистки {variant.name}:', df_clean['target'].mean())
            X_TRAIN, X_VAL, X_TEST, X_CROSS = split_ds(df_clean)
            save_dataset(X_TRAIN, Path(variant.name).joinpath(target.name), 'train')
            save_dataset(X_VAL, Path(variant.name).joinpath(target.name), 'val')
            save_dataset(X_TEST, Path(variant.name).joinpath(target.name), 'test')
            save_dataset(X_CROSS, Path(variant.name).joinpath(target.name), 'cross')
