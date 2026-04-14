import os
from typing import List, Optional

import joblib
import numpy as np

from db import get_db_connection
from training import training
from utils.constants import MODEL_CACHE_PATH, KMEANS_N_CLUSTERS

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None
    print("[kmeans_training] Aviso: sklearn não está instalado.")

KMEANS_MODEL_FILE = os.path.join(MODEL_CACHE_PATH, "kmeans_model.pkl")


def train_kmeans(n_clusters: int = KMEANS_N_CLUSTERS, random_state: int = 42) -> dict:
    if KMeans is None:
        raise RuntimeError("sklearn não encontrado. Instale com: pip install scikit-learn")

    training._load_cache_if_needed()
    item_df = training._item_df
    if item_df is None:
        item_df = training.load_items_from_db()
        if item_df is None or item_df.empty:
            raise RuntimeError("Nenhum item disponível para treinar K-Means.")

    feature_cols = [c for c in item_df.columns if c != 'id']
    X = item_df[feature_cols].values.astype(float)

    # Garante que n_clusters não supera o número de itens
    n_clusters = min(n_clusters, len(X))

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)

    model_data = {
        "model": model,
        "labels": labels,
        "item_ids": item_df['id'].tolist(),
    }

    os.makedirs(MODEL_CACHE_PATH, exist_ok=True)
    joblib.dump(model_data, KMEANS_MODEL_FILE)
    print(f"[kmeans_training] Modelo salvo em: {KMEANS_MODEL_FILE}. "
          f"Clusters: {n_clusters}, Inertia: {model.inertia_:.2f}")
    return {"n_clusters": n_clusters, "inertia": float(model.inertia_)}


def load_kmeans_model() -> Optional[dict]:
    if not os.path.exists(KMEANS_MODEL_FILE):
        print(f"[kmeans_training] Modelo não encontrado em: {KMEANS_MODEL_FILE}")
        return None
    try:
        data = joblib.load(KMEANS_MODEL_FILE)
        print(f"[kmeans_training] Modelo carregado de: {KMEANS_MODEL_FILE}")
        return data
    except Exception as e:
        print(f"[kmeans_training] Erro ao carregar modelo: {e}")
        return None


def recommend_with_kmeans(user_id: str, count: int = 10) -> List[str]:
    """
    Recomenda itens do mesmo cluster que os itens com os quais o usuário interagiu.
    Clusters com mais interações (ponderadas) têm prioridade.
    """
    training._load_cache_if_needed()
    if training._item_df is None:
        return (training._popularity or [])[:count]

    model_data = load_kmeans_model()
    if model_data is None:
        print("[kmeans_training] Modelo K-Means indisponível — usando fallback de popularidade.")
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

    # Mapeia item_id -> cluster
    id_to_cluster = {
        iid: label
        for iid, label in zip(model_data["item_ids"], model_data["labels"])
    }

    # Acumula peso por cluster com base nas interações do usuário
    cluster_weights: dict = {}
    interacted_set = set()
    for it in user_interactions:
        iid = it["jewelry_id"]
        interacted_set.add(iid)
        cluster = id_to_cluster.get(iid)
        if cluster is not None:
            weight = training.interaction_score(it)
            cluster_weights[cluster] = cluster_weights.get(cluster, 0.0) + weight

    if not cluster_weights:
        return (training._popularity or [])[:count]

    # Pontua candidatos pelo peso do cluster ao qual pertencem
    candidate_scores = {}
    for iid in model_data["item_ids"]:
        if iid in interacted_set:
            continue
        cluster = id_to_cluster.get(iid)
        if cluster in cluster_weights:
            candidate_scores[iid] = cluster_weights[cluster]

    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    recommended = [iid for iid, _ in sorted_candidates][:count]

    # Fallback por popularidade
    if len(recommended) < count:
        for pid in (training._popularity or []):
            if pid not in recommended and pid not in interacted_set:
                recommended.append(pid)
            if len(recommended) >= count:
                break

    return recommended[:count]