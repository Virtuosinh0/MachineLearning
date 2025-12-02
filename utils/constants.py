import os

MODEL_CACHE_PATH = "models"
XGB_MODEL_PATH = "model_cache"
ITEM_FEATURES_PATH = os.path.join(MODEL_CACHE_PATH, "item_features.pkl")
POPULARITY_PATH = os.path.join(MODEL_CACHE_PATH, "popularity.pkl")

TYPE_WEIGHTS = {
    "ignored": -0.1,
    "click": 1.0,
    "shopping_cart": 3.0,
    "bought": 5.0
}

HALF_LIFE_DAYS = 30.0

PRICE_WEIGHT = 2.0