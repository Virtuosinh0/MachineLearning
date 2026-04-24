import os
from typing import List, Optional

import joblib
import numpy as np

from db import get_db_connection
from training import training
from utils.constants import MODEL_CACHE_PATH, KNN_N_NEIGHBORS

try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    NearestNeighbors = None
    print("[knn_training] Aviso: sklearn não está instalado.")

KNN_MODEL_FILE = os.path.join(MODEL_CACHE_PATH, "knn_model.pkl")


def train_knn(n_neighbors: int = KNN_N_NEIGHBORS, metric: str = "cosine") -> dict:
    if NearestNeighbors is None:
        raise RuntimeError("sklearn não encontrado. Instale com: pip install scikit-learn")

    training._load_cache_if_needed()
    item_df = training._item_df
    if item_df is None:
        item_df = training.load_items_from_db()
        if item_df is None or item_df.empty:
            raise RuntimeError("Nenhum item disponível para treinar KNN.")

    feature_cols = [c for c in item_df.columns if c != 'id']
    X = item_df[feature_cols].values.astype(float)

    # +1 porque o próprio item é retornado como vizinho mais próximo
    n_neighbors = min(n_neighbors + 1, len(X))

    model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm="brute", n_jobs=-1)
    model.fit(X)

    # Mean neighbor distance (excludes self, which is always index 0)
    distances, _ = model.kneighbors(X)
    mean_neighbor_dist = float(np.mean(distances[:, 1:]))

    model_data = {
        "model": model,
        "item_ids": item_df['id'].tolist(),
        "X": X,
        "metric": metric,
    }

    os.makedirs(MODEL_CACHE_PATH, exist_ok=True)
    joblib.dump(model_data, KNN_MODEL_FILE)
    print(f"[knn_training] Modelo salvo em: {KNN_MODEL_FILE}. "
          f"Vizinhos: {n_neighbors - 1}, Métrica: {metric}, "
          f"Distância média entre vizinhos: {mean_neighbor_dist:.4f} (menor=itens mais similares)")
    return {"n_neighbors": n_neighbors - 1, "metric": metric, "mean_neighbor_dist": mean_neighbor_dist}


def load_knn_model() -> Optional[dict]:
    if not os.path.exists(KNN_MODEL_FILE):
        print(f"[knn_training] Modelo não encontrado em: {KNN_MODEL_FILE}")
        return None
    try:
        data = joblib.load(KNN_MODEL_FILE)
        print(f"[knn_training] Modelo carregado de: {KNN_MODEL_FILE}")
        return data
    except Exception as e:
        print(f"[knn_training] Erro ao carregar modelo: {e}")
        return None


def recommend_with_knn(user_id: str, count: int = 10) -> List[str]:
    """
    Para cada item com o qual o usuário interagiu, busca os K vizinhos mais próximos
    no espaço de features. Vizinhos acumulam score = similaridade * peso_da_interação.
    """
    training._load_cache_if_needed()
    if training._item_df is None:
        return (training._popularity or [])[:count]

    model_data = load_knn_model()
    if model_data is None:
        print("[knn_training] Modelo KNN indisponível — usando fallback de popularidade.")
        return (training._popularity or [])[:count]

    conn = get_db_connection()
    if conn is None:
        return (training._popularity or [])[:count]

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT jewelry_id, type, created_at FROM user_interaction "
                "WHERE user_id = %s ORDER BY created_at DESC LIMIT 50;",
                (user_id,)
            )
            rows = cur.fetchall()
            user_interactions = [
                {"jewelry_id": str(r[0]), "type": r[1], "created_at": r[2]}
                for r in rows
            ]
    finally:
        conn.close()

    if not user_interactions:
        return (training._popularity or [])[:count]

    id_to_idx = {iid: idx for idx, iid in enumerate(model_data["item_ids"])}
    interacted_set = {it["jewelry_id"] for it in user_interactions}
    candidate_scores: dict = {}

    for it in user_interactions:
        iid = it["jewelry_id"]
        if iid not in id_to_idx:
            continue

        weight = training.interaction_score(it)
        item_vector = model_data["X"][id_to_idx[iid]].reshape(1, -1)
        distances, indices = model_data["model"].kneighbors(item_vector)

        for dist, neighbor_idx in zip(distances[0], indices[0]):
            neighbor_id = model_data["item_ids"][neighbor_idx]
            if neighbor_id == iid:
                continue
            # Para métrica cosine: distância ∈ [0, 2], similaridade = 1 - distância
            similarity = 1.0 - dist
            candidate_scores[neighbor_id] = (
                candidate_scores.get(neighbor_id, 0.0) + similarity * weight
            )

    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    recommended = [iid for iid, _ in sorted_candidates if iid not in interacted_set][:count]

    # Fallback por popularidade
    if len(recommended) < count:
        for pid in (training._popularity or []):
            if pid not in recommended and pid not in interacted_set:
                recommended.append(pid)
            if len(recommended) >= count:
                break

    return recommended[:count]