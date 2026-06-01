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

HALF_LIFE_DAYS = 120.0

PRICE_WEIGHT = 2.0

KMEANS_N_CLUSTERS = 10
KNN_N_NEIGHBORS = 14

# Reciprocal Rank Fusion — constante de amortecimento (Cormack et al., 2009)
RRF_K = 60

# Pesos do Reciprocal Rank Fusion no sistema híbrido (altere aqui para experimentar)
RRF_WEIGHT_CB     = 1.0   # Content-Based (similaridade de features)
RRF_WEIGHT_KNN    = 1.0   # KNN (vizinhos mais próximos)
RRF_WEIGHT_SVD    = 1.0   # SVD (filtragem colaborativa)
RRF_WEIGHT_KMEANS = 1.0   # K-Means (afinidade por cluster)

# Threshold mínimo de similaridade cosseno para o Content-Based (seção 3.3 do artigo)
CB_SIMILARITY_THRESHOLD = 0.1

# Número de fatores latentes do SVD (seção 3.3 do artigo)
SVD_N_FACTORS = 30

# Flags de grid search — True: executa varredura na validação / False: usa o valor fixo acima
GRID_SEARCH_CB  = False
GRID_SEARCH_KNN = False
GRID_SEARCH_SVD = False