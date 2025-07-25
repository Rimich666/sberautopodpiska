# Этапы большого пути
## Этап первый (давно забытый)
**Сушка данных, набор фичей**
```bash
    prepare_session.py
    prepare_session.ipymb 
```
- Как сушили данные писать не буду
- Фичей добавили:
  1. is_returning - повторный визит
  2. brand_tier - категория бренда девайса
  3. frequent_visitor - частый визитёр
  4. visit_month - месяц визита
  5. visit_season - сезон визита
  6. visit_day_week - день недели
  7. is_weekend - выходной
  8. visit_hour - час визита
  9. is_peak_hour - час пик
  10. time_of_day - время дня
  11. has_utm_keyword - бинарник, есть ключ
  12. has_utm_campaign - бинарник, есть кампания
```python
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
```
## Этап второй (драматический)
**Отбор фичей**
```bash
  feature_selection.py
  learn.ipynb
```
Название "learn.ipynb" вводит в заблуждение, но отражает историю процесса, это артефакт.
Изначально отбор был по ```ROC-AUC```, потом по ```F1```.
Сейчас, задним умом понятно, что особой разницы нет, 
но тогда - это казалось выходом из ситуации тяжёлого дисбаланса классов.
Отбор был в четырёх вариантах, с разными ```target```, и разной степенью очистки.
В последствии, как то само собой остался один - рабочий вариант. 
С ```target``` из восьми событий и с ```тяжёлой очисткой```

Best AUC = 0.7107507084307269
Best features 
```
[ 'utm_campaign',
  'visit_month', 
  'utm_source', 
  'visit_number', 
  'device_screen_resolution', 
  'geo_city', 
  'device_browser', 
  'time_of_day', 
  'device_brand'
  ]
```
- ROC-AUC: 0.7108

📝 Classification Report:

|              |precision |   recall | f1-score | support  |
|--------------|----------|----------|----------|----------|
|0             |0.984301  |0.633569  |0.770918  | 333203   |
|1             |0.051321  |0.662355  |0.095260  | 9972     |
|accuracy      |0.634405  |0.634405  |0.634405  | 0.634405 |
|macro avg     |0.517811  |0.647962  |0.433089  | 343175   |
|weighted avg  |0.957191  |0.634405  |0.751285  | 343175   |

Вот его результат на этапе отбора.
Опять же, задним умом понятно - надо было остановится.

Тем не менее отбор по F1 был сделан:

Best features: 
```
[
  'visit_month', 
  'utm_medium', 
  'utm_campaign', 
  'utm_keyword', 
  'utm_source'
]
```
- Best F1_1: 0.08561635140281833
- ROC-AUC: 0.6812898205412515
- PR-AUC: 0.06278089277916321

📊 Отчёт по метрикам:

|   |precision  |  recall  |f1-score|
|---|-----------|----------|--------|
|0   |0.983498  |0.577363  |0.727592|
|1   |0.045701  |0.676294  |0.085616|

Этот вариант был принят за базу. Может и зря.
Только сейчас вижу, метрики то похуже. Но посыпание головы пеплом - это не наш стиль.

Да чуть не забыл. Вот окончательный вариант вызова процедуры подбора:
```python
    feature_selection(MetricNames.f1_1, 0.0001)
```
А вот и сигнатура:
```python
def feature_selection(target_metric: MetricNames.auc, min_improvement: float = 0.001):
```

## Этап третий (занудный)
#### Отчаянная попытка побороть дисбаланс с помощью подбора гиперов с семплированием
```bash
  hiper_parameters.ipynb
  selection_hyper.py
  oversampling.py
```
**Это вызов:**
```python
hyper_select(model=models.catboost, metric=MetricNames.f1_1, trial_count=50, features_metric=MetricNames.f1_1, parts=(0,))
```
**Это сигнатура:**
```python
def hyper_select(
        model: Model = models.catboost,
        metric: MetricNames = DEFAULT_METRIC,
        trial_count=TRIALS,
        features_metric: MetricNames = None,
        parts=MINOR_PARTS):
    """
        Принимает модель,
        Метрику для оптимизации,
        количество триалов,
        метрику отбора фичей, для подгрузки этих самых фмчей
        долю, которую хочет занять минорный класс в датасете
    """
```
#### Результаты для CatBoost:
- Model: <catboost.core.CatBoostClassifier object at 0x00000275AFEF2A80>
- Все параметры из study.best_params: ['depth', 'lr', 'iterations', 'l2']

🔍 Лучшие параметры для target: sub8, variant: hard, parts count: 0
- Метрика отбора фичей: f1_1
- Метрика оптимизации: f1_1
  - depth: 6
  - lr: 0.14703266835718107
  - iterations: 645
  - l2: 3

- Оптимальный порог: 0.6653868128753974

**📊 Значимость фичей из лучшей модели:**

| feature      |importance|
|--------------|----------|
| utm_source   | 28.965755|
| utm_campaign | 20.647970|
| isit_month   | 19.541513|
| utm_medium   | 17.075351|
| utm_keyword  | 13.769410|

**🔍 Подробные метрики лучшей модели:**

📊 Classification Report:

|              | precision | recall |  f1-score| support |
|--------------|-----------|--------|----------|---------|
|     Class 0  | 0.98      | 0.59   |      0.74| 333203  |
|     Class 1  | 0.05      | 0.66   |      0.09| 9972    |
|    accuracy  |           |        |      0.60| 343175  |
|   macro avg  |  0.51     | 0.63   |      0.41| 343175  |
|weighted avg  | 0.96      | 0.60   |     0.72 | 343175  |


📝 Classification Report с оптимизированным порогом: 0.6653868128753974:

|              | precision |    recall|  f1-score|   support|
|--------------|-----------|----------|----------|----------|
| Class 0      | 0.98      |      0.91|      0.94|    333203|
| Class 1      | 0.07      |      0.24|      0.11|      9972|
| accuracy     |           |          |      0.89|    343175|
| macro avg    |       0.53|      0.58|      0.53|    343175|
| weighted avg |       0.95|      0.89|      0.92|    343175|

📈 ROC-AUC Score: 0.6795451856212482

#### Результаты для LGBMClassifier:
- Model: LGBMClassifier(class_weight='balanced', device='gpu',
               learning_rate=0.010002311589549463, max_depth=4, metric='f1',
               n_estimators=843, n_jobs=-1, objective='binary', random_state=42,
               reg_lambda=1, verbosity=-1)
- Все параметры из study.best_params: ['max_depth', 'learning_rate', 'n_estimators', 'reg_lambda']

**🔍 Лучшие параметры для target: sub8, variant: hard, parts count: 0**
- Метрика отбора фичей: f1_1
- Метрика оптимизации: f1_1
  - max_depth: 4
  - learning_rate: 0.010002311589549463
  - n_estimators: 843
  - reg_lambda: 1

Оптимальный порог: 0.6624217241647351

**📊 Значимость фичей из лучшей модели:**

| feature       | importance|
|---------------|-----------|
| utm_campaign  |       3941|
| utm_source    |       2803|
| visit_month   |       1968|
| utm_keyword   |       1783|
| utm_medium    |       1540|

**🔍 Подробные метрики лучшей модели:**

📊 Classification Report:

|              |precision |   recall | f1-score |  support|
|--------------|----------|----------|----------|---------|
| Class 0      |     0.98 |     0.56 |     0.71 |   333203|
| Class 1      |      0.04|      0.68|      0.08|     9972|
| accuracy     |          |          |      0.56|   343175|
| macro avg    |      0.51|      0.62|      0.40|   343175|
| weighted avg |      0.96|      0.56|      0.70|   343175|


📝 Classification Report с оптимизированным порогом: 0.6624217241647351:

|              |precision|    recall|  f1-score|   support|
|--------------|---------|----------|----------|----------|
| Class 0      |     0.97|      0.95|      0.96|    333203|
| Class 1      |     0.08|      0.15|      0.11|      9972|
| accuracy     |         |          |      0.93|    343175|
| macro avg    |     0.53|      0.55|      0.53|    343175|
| weighted avg |     0.95|      0.93|      0.94|    343175|

📈 ROC-AUC Score: 0.6757920403436106

Все метрики приведены с прогона без семплтрования. 
Прогоны с семплированием не дали никакого изменения результата, ни в плюс, ни в минус
А вообще, по сравнению со стартовыми значениями на изменилось **ничего.**

## Этап четвёртый (весёлый)
#### ~~Сгорел сарай, гори и хата~~ Пробуем ансамбли и другую тяжёлую артиллерию

### Калибр 1
#### Stacking
```bash
  final.ipynb
  ensemble.py
```
Название ```final.ipynb``` - это тоже исторический артефакт, 
хотя ближе по смыслу к содержанию, чем предыдущий

- Решающая модель: LogisticRegression
- База: LGBMClassifier, CatBoost
Борьба с дисбалансом (куда уж без неё) свелась к ручной установке весов классов:

```python

 self.stacking_model = LogisticRegression(
        class_weight={0: 1, 1: 33},
        C=0.1,
        solver='lbfgs',
        max_iter=1000
    )
```
Без борьбы с дисбалансом LR вообще отказалась распознавать class1, вплоть до
Zero Division

Да, метрики:

**🧪 Тестирование Stacking:**
- 📊 ROC-AUC: 0.6743
- 📌 Оптимальный порог: 0.6287

📝 Classification Report (порог 0.5):

|              |precision  |  recall | f1-score |  support|
|--------------|-----------|---------|----------|---------|
|     Class 0  |     0.98  |    0.60 |     0.74 |   333203|
|     Class 1  |     0.05  |    0.64 |     0.09 |     9972|
|    accuracy  |           |         |     0.60 |   343175|
|   macro avg  |     0.51  |    0.62 |     0.41 |   343175|
|weighted avg  |     0.96  |    0.60 |     0.73 |   343175|


📝 Classification Report (оптимальный порог):

|              |precision |   recall  |f1-score  | support|
|--------------|----------|-----------|----------|--------|
|     Class 0  |     0.98 |     0.89  |    0.93  |  333203|
|     Class 1  |     0.07 |     0.27  |    0.11  |    9972|
|    accuracy  |          |           |    0.88  |  343175|
|   macro avg  |     0.52 |     0.58  |    0.52  |  343175|
|weighted avg  |     0.95 |     0.88  |    0.91  |  343175|

Не сдвинулись ни на милиметр.

### Калибр 2
#### LightAutoMLModel
```bash
  light_auto_ml.py
  light_auto_ml.ipynb
```
Из всего предлагаемого компота мы использовали две базовы модели
- LGBMClassifier
- CatBoost
Точнее их бустовые варианты, что отразилось в передаче параметров.

Это самый скучный эксперимент потому сразу к метрикам:

|           | precision | recall  | f1-score | support  |
|-----------|-----------|---------|----------|----------|
|Class 0    | 0.9730    | 0.9416  | 0.9570   | 333203.0 |
|Class 1    | 0.0610    | 0.1268  | 0.0824   | 9972.0   |

- Accuracy: 0.9179
- ROC-AUC: 0.5810
- Macro F1: 0.5197

Ну, как видно, всё совсем не хорошо.

## Этап последний (радостный)
#### И не потому что последний, просто написать микробэк на FastApi и микрофронт на ванили, какое счастье!

Демка написана давно, тогда, когда надо было всё это прекратить.
Модель там старенькая, по метрикам самая плохая - первая.
Ещё на старом наборе фичей.
При старте грузит модель и ждёт фронта.
Фронт просит все уникальные значения категорий.
Заполняет списки в соответствующих полях.
Имеет три кнопки. 
- Заполнить случайными значениями.
- Запросить предсказание
- Запрашивать автоматически

Вторая кнопка устарела. При заполнении идёт запрос на предсказание.
Автомат перебирает случайные значения фичей и запрашивает предсказания, пока не будет превышен порог.
Ну вот так не замысловато.
Была мысль переделать. Но я решил от этой мысли отказаться.

## Запуск приложения

```bash
python main.py
```

```bash
(base) PS F:\Projects\tgu\hahaton25_2\sberautopodpiska> python main.py
2025-07-25 08:22:29 | DEBUG | Using proactor: IocpProactor
INFO:     Started server process [420]
INFO:     Waiting for application startup.
INFO:     Waiting for application startup.
2025-07-25 08:22:29 | INFO | F:\Projects\tgu\hahaton25_2\sberautopodpiska\src
2025-07-25 08:22:29 | INFO | F:\Projects\tgu\hahaton25_2\sberautopodpiska\src\base

```
**На порту 8088 появится бэк.**

**Вот этот файлик запустить в браузере ```demo/index.html```**

**Запустится фронт**

Запускаемую модель я перенёс в ```src```, забыл что все модели не гитуются, извините.


# Оду отрицательному результату просто необходимо закончить так:

> Результат: ноль.  
> Плюс две бессонные недели.  
> Вывод: **beer time**.

