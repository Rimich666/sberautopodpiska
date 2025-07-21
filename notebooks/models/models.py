from notebooks.models.catboost_model import CatBoost
from notebooks.models.lightgbm_model import LightGBM


class Models:
    def __init__(self):
        self.catboost = CatBoost()
        self.lightgbm = LightGBM()


models = Models()
