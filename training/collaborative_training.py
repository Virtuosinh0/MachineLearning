import pandas as pd
import numpy as np
import joblib
import os
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from db import get_db_connection
from training import training
from utils.constants import MODEL_CACHE_PATH

SVD_MODEL_FILE = os.path.join(MODEL_CACHE_PATH, "svd_model.pkl")

def train_collaborative_svd():
    print("[svd_training] Iniciando treino de Filtragem Colaborativa...")
    conn = get_db_connection()
    if not conn: return
    
    try:
        # Busca todas as interações para criar a matriz usuário-item
        df = pd.read_sql_query("SELECT user_id, jewelry_id, type, created_at FROM user_interaction", conn)
        if df.empty: return
        
        df['score'] = df.apply(training.interaction_score, axis=1)
        
        # Cria matriz esparsa
        user_codes = df['user_id'].astype('category')
        item_codes = df['jewelry_id'].astype('category')
        
        pivot_matrix = csr_matrix((df['score'], (user_codes.cat.codes, item_codes.cat.codes)))
        
        # Fatorização (k=faixas latentes)
        k = min(20, pivot_matrix.shape[1] - 1)
        u, sigma, vt = svds(pivot_matrix.asfptype(), k=k)
        sigma = np.diag(sigma)
        
        # Predição latente: U * Sigma * Vt
        predicted_ratings = np.dot(np.dot(u, sigma), vt)
        
        model_data = {
            "ratings": predicted_ratings,
            "user_map": user_codes.cat.categories,
            "item_map": item_codes.cat.categories
        }
        
        os.makedirs(MODEL_CACHE_PATH, exist_ok=True)
        joblib.dump(model_data, SVD_MODEL_FILE)
        print(f"[svd_training] Modelo salvo em: {SVD_MODEL_FILE}")
    finally:
        conn.close()

def recommend_svd(user_id: str, count: int = 10):
    if not os.path.exists(SVD_MODEL_FILE): return []
    
    data = joblib.load(SVD_MODEL_FILE)
    if user_id not in data["user_map"]: return []
    
    u_idx = list(data["user_map"]).index(user_id)
    u_ratings = data["ratings"][u_idx]
    
    # Ordena itens por score latente
    item_indices = np.argsort(u_ratings)[::-1]
    recs = [data["item_map"][i] for i in item_indices[:count]]
    return recs