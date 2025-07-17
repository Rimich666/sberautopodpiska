import gc
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from tqdm import tqdm
from src.logger import logger

BATCH_SIZE = 200000


def preprocess_data(X, categorical_features):
    # Кодируем категориальные признаки
    from sklearn.preprocessing import OrdinalEncoder

    cat_cols = [X.columns[i] for i in categorical_features]

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[cat_cols] = encoder.fit_transform(X[cat_cols])

    return X, cat_cols, encoder


def postprocess_data(X, cat_cols, encoder):
    # Декодируем обратно в строки
    X[cat_cols] = encoder.inverse_transform(X[cat_cols])
    return X


def batch_smote(X, y, categorical_features, percent, batch_size=50000):
    """
    Батчевая обработка SMOTE для больших датасетов
    """
    # Сбрасываем индексы для гарантии совпадения
    X = X.reset_index(drop=True).copy()
    y = y.reset_index(drop=True).copy()

    # Разделяем классы с проверкой индексов
    minor_mask = (y == 1).to_numpy()
    X_minor = X.loc[minor_mask].copy()
    X_major = X.loc[~minor_mask].copy()

    # Вычисляем сколько нужно сгенерировать
    n_to_generate = int(len(X_major) * percent / (100 - percent)) - len(X_minor)

    synthetic_samples = []
    n_batches = max(1, int(np.ceil(n_to_generate / batch_size)))

    logger.debug(f"Начало батчевой обработки. Будет {n_batches} батчей по ~{batch_size} сэмплов")

    for batch_idx in tqdm(range(n_batches), desc="SMOTE батчи"):
        current_batch_size = min(batch_size, n_to_generate - batch_idx * batch_size)

        # Берем подвыборку минорного класса
        batch_minor = X_minor.sample(
            n=min(len(X_minor), 5000),  # Ограничиваем размер для k-neighbors
            random_state=42 + batch_idx
        ).copy()

        batch_major = X_major.sample(
            n=min(10000, len(X_major)),
            random_state=42 + batch_idx
        ).copy()

        # Создаем временный датасет для SMOTE
        temp_X = pd.concat([batch_major, batch_minor])
        temp_y = pd.Series(
            [0] * len(batch_major) + [1] * len(batch_minor),
            index=temp_X.index  # Критически важно!
        )

        smote = SMOTENC(
            categorical_features=categorical_features,
            sampling_strategy={1: current_batch_size + len(batch_minor)},
            k_neighbors=min(5, len(batch_minor) - 1),
            random_state=42 + batch_idx
        )

        try:
            X_res, _ = smote.fit_resample(temp_X, temp_y)
            synthetic_samples.append(X_res[len(temp_X[temp_y == 0]):])
        except Exception as e:
            logger.warning(f"Ошибка в батче {batch_idx}: {str(e)}")
            continue

        gc.collect()

    logger.debug(f"Собираем результат")
    if synthetic_samples:
        X_resampled = pd.concat([X_major] + [X_minor] + synthetic_samples)
        y_resampled = pd.Series(
            [0] * len(X_major) + [1] * (len(X_minor) + sum(len(b) for b in synthetic_samples))
        )
        return X_resampled, y_resampled
    return X, y


def standard_smote(X, y, categorical_features, percent):
    """Стандартный SMOTE с обработкой ошибок"""
    try:
        # X_processed, cat_cols, encoder = preprocess_data(X.copy(), categorical_features)
        # logger.debug(f"Размер данных после приведения: {X_processed.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
        target_counts = y.value_counts()
        desired_minority = int(target_counts[0] * percent / (100 - percent))

        smote = SMOTENC(
            categorical_features=categorical_features,
            sampling_strategy={1: desired_minority},
            random_state=42
        )
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res
    except Exception as e:
        logger.error(f"Ошибка в standard_smote: {str(e)}")
        return X, y


def oversample(X, y, categorical_features, percent):
    logger.debug(f"Реальный размер данных: {X.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
    target_counts = y.value_counts()
    current_minority = target_counts.get(1, 0)
    majority = target_counts.get(0, 0)
    desired_minority = int(majority / (100 - percent) * percent)
    n_to_generate = desired_minority - current_minority

    if current_minority < desired_minority:
        if n_to_generate > BATCH_SIZE:
            logger.warning(f"Надо сгенерировать {n_to_generate} строк, будем батчить")
            X_res, y_res = batch_smote(X, y, categorical_features, percent, BATCH_SIZE)
            print(f"Новое распределение классов:\n", pd.Series(y_res).value_counts())
            return X_res, y_res
        X_res, y_res = standard_smote(X, y, categorical_features, percent)
        print("Новое распределение классов:\n", pd.Series(y_res).value_counts())
        return X_res, y_res

    print("\nOversampling не требуется: желаемый процент уже достигнут")
    return X, y
