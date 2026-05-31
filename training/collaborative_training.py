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
        df['user_id'] = df['user_id'].astype(str)
        df['jewelry_id'] = df['jewelry_id'].astype(str)

        # Cria matriz esparsa
        user_codes = df['user_id'].astype('category')
        item_codes = df['jewelry_id'].astype('category')
        
        pivot_matrix = csr_matrix((df['score'], (user_codes.cat.codes, item_codes.cat.codes)))
        
        # Fatorização (k=faixas latentes)
        k = min(20, pivot_matrix.shape[1] - 1)
        u, sigma_vals, vt = svds(pivot_matrix.asfptype(), k=k)
        sigma = np.diag(sigma_vals)

        # Predição latente: U * Sigma * Vt
        predicted_ratings = np.dot(np.dot(u, sigma), vt)

        # Variância explicada pelos k fatores latentes
        total_variance = float(pivot_matrix.power(2).sum())
        explained_variance = float(np.sum(sigma_vals ** 2)) / total_variance if total_variance > 0 else 0.0

        model_data = {
            "ratings": predicted_ratings,
            "user_map": user_codes.cat.categories,
            "item_map": item_codes.cat.categories
        }

        os.makedirs(MODEL_CACHE_PATH, exist_ok=True)
        joblib.dump(model_data, SVD_MODEL_FILE)
        print(f"[svd_training] Modelo salvo em: {SVD_MODEL_FILE}. "
              f"Fatores latentes: {k}, Variância explicada: {explained_variance:.1%}")
    finally:
        conn.close()

def recommend_svd(user_id: str, count: int = 10):
    if not os.path.exists(SVD_MODEL_FILE): return []

    data = joblib.load(SVD_MODEL_FILE)

    print(f"\n--- [MÉTRICAS] SVD (Filtragem Colaborativa) ---")
    if user_id not in data["user_map"]:
        print(f"  [Usuário no modelo] Não — sem histórico colaborativo; retornando vazio")
        print(f"------------------------------------------------\n")
        return []

    conn = get_db_connection()
    interacted_set = set()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT jewelry_id FROM user_interaction WHERE user_id = %s;",
                    (user_id,)
                )
                interacted_set = {str(r[0]) for r in cur.fetchall()}
        finally:
            conn.close()

    u_idx = list(data["user_map"]).index(user_id)
    u_ratings = data["ratings"][u_idx]

    item_indices = np.argsort(u_ratings)[::-1]
    recs = [
        str(data["item_map"][i])
        for i in item_indices
        if str(data["item_map"][i]) not in interacted_set
    ][:count]

    top_scores = u_ratings[item_indices[:count]]
    print(f"  [Usuário no modelo]  Sim (índice {u_idx} de {len(data['user_map'])} usuários)")
    print(f"  [Itens no modelo]    {len(data['item_map'])} itens com scores latentes")
    print(f"  [Interações filtradas] {len(interacted_set)} itens removidos do ranking")
    print(f"  [Score latente top-1]    {float(top_scores[0]):.4f}")
    print(f"  [Score latente top-{count}]   {float(top_scores[-1]):.4f}")
    print(f"  [Score médio top-{count}]     {float(np.mean(top_scores)):.4f}")
    print(f"------------------------------------------------\n")

    return recs