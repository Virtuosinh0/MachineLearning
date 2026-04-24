import numpy as np
from db import get_db_connection
from training.training import _load_cache_if_needed, _popularity, _item_df, _similarity_matrix, interaction_score
from training import training
from training.collaborative_training import recommend_svd
from training.kmeans_training import recommend_with_kmeans
from training.knn_training import recommend_with_knn

def get_hybrid_recommendations(user_id: str, count: int = 10):
    """
    Combina Content-Based, SVD, K-Means e KNN via Reciprocal Rank Fusion.

    Pesos:
      - Content-Based : 1.0  (similaridade por features do item)
      - KNN           : 0.9  (vizinhos mais próximos no espaço de features)
      - SVD           : 0.8  (comportamento coletivo / filtragem colaborativa)
      - K-Means       : 0.6  (afinidade por cluster)
    """
    pool = count * 2

    cb_recs    = get_recommendations(user_id, count=pool)
    knn_recs   = recommend_with_knn(user_id, count=pool)
    svd_recs   = recommend_svd(user_id, count=pool)
    kmeans_recs = recommend_with_kmeans(user_id, count=pool)

    scores = {}

    def apply_score(recs, weight):
        for i, iid in enumerate(recs):
            score = (1.0 / (i + 1)) * weight
            scores[iid] = scores.get(iid, 0.0) + score

    apply_score(cb_recs,     1.0)
    apply_score(knn_recs,    0.9)
    apply_score(svd_recs,    0.8)
    apply_score(kmeans_recs, 0.6)

    sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    final_recs = [str(item[0]) for item in sorted_recs[:count]]

    # --- Métricas de recomendação ---
    print(f"\n--- [MÉTRICAS] AVALIAÇÃO DA RECOMENDAÇÃO HÍBRIDA ---")

    # 1. Contribuição dos algoritmos
    print(f"  [Contribuição] CB={len(cb_recs)}, KNN={len(knn_recs)}, "
          f"SVD={len(svd_recs)}, K-Means={len(kmeans_recs)}, "
          f"Candidatos únicos (RRF)={len(scores)}")

    # 2. Cobertura do catálogo
    total_items = len(training._item_df) if training._item_df is not None else None
    if total_items:
        coverage = len(scores) / total_items
        print(f"  [Cobertura] {coverage:.1%} do catálogo coberto ({len(scores)}/{total_items} itens)")

    # 3. Diversidade intra-lista (média das distâncias cosseno par-a-par)
    if training._item_df is not None and len(final_recs) >= 2:
        id_to_idx = {iid: idx for idx, iid in enumerate(training._item_df['id'].tolist())}
        feature_cols = [c for c in training._item_df.columns if c != 'id']
        indices = [id_to_idx[iid] for iid in final_recs if iid in id_to_idx]
        if len(indices) >= 2:
            vecs = training._item_df[feature_cols].values[indices].astype(float)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            vecs_norm = vecs / norms
            sim_matrix = np.dot(vecs_norm, vecs_norm.T)
            n = len(indices)
            upper_tri = sim_matrix[np.triu_indices(n, k=1)]
            diversity = float(1.0 - np.mean(upper_tri))
            print(f"  [Diversidade intra-lista] {diversity:.4f} (0=sem diversidade, 1=máxima diversidade)")

    print(f"----------------------------------------------------\n")

    return final_recs

def get_recommendations(user_id: str, count: int = 10):
    
    training._load_cache_if_needed()
    
    if training._item_df is None or training._similarity_matrix is None:
        print("[recommendation_logic] Modelo não preparado. Chame train_model() primeiro.")
        return []
        
    conn = get_db_connection()
    if conn is None: return []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT jewelry_id, type, created_at FROM user_interaction WHERE user_id = %s ORDER BY created_at DESC LIMIT 50;", (user_id,))
            rows = cur.fetchall()
            user_interactions = [{"jewelry_id": str(r[0]), "type": r[1], "created_at": r[2]} for r in rows]
    finally:
        conn.close()

    if not user_interactions:
        print(f"[recommendation_logic] Usuário {user_id} sem interações: usando fallback por popularidade.")
        return (training._popularity[:count] if training._popularity else [])[:count]

    seed_scores = {}
    print("\n--- [DEBUG] DADOS BRUTOS DA INTERAÇÃO SENDO PROCESSADOS ---")
    for it in user_interactions:
        print(f"  - Lendo: jewelry_id={it['jewelry_id']}, type='{it['type']}', created_at={it['created_at']}")
        sid = it['jewelry_id']
        s = training.interaction_score(it)
        seed_scores[sid] = seed_scores.get(sid, 0.0) + s
    
    print("\n--- [DEBUG] SCORES DE INTERAÇÃO (SEEDS) ---")
    if not seed_scores: print("  - Nenhum score de seed calculado.")
    for item_id, score in seed_scores.items():
        print(f"  - Item (seed): {item_id}, Score: {score:.4f}")

    id_to_idx = {iid: idx for idx, iid in enumerate(training._item_df['id'].tolist())}
    candidate_scores = {}
    for seed_id, weight in seed_scores.items():
        if seed_id not in id_to_idx: continue
        sidx = id_to_idx[seed_id]
        sims = training._similarity_matrix[sidx]
        for idx, sim_val in enumerate(sims):
            iid = training._item_df['id'].iloc[idx]
            if iid == seed_id: continue
            candidate_scores[iid] = candidate_scores.get(iid, 0.0) + sim_val * weight

    interacted_set = {it['jewelry_id'] for it in user_interactions}
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\n--- [DEBUG] SCORES FINAIS DOS CANDIDATOS (TOP 20) ---")
    if not sorted_candidates: print("  - Nenhum score de candidato calculado.")
    for item_id, score in sorted_candidates[:20]:
        print(f"  - Item (candidato): {item_id}, Score Final: {score:.4f}")
    
    print("\n--- [DIAGNÓSTICO] ANÁLISE DE SIMILARIDADE DOS TOP 3 CANDIDATOS ---")
    if seed_scores:
        most_influential_seed = max(seed_scores, key=seed_scores.get)
        print(f"Analisando similaridade com o item mais influente do seu histórico: {most_influential_seed}")
        
        for candidate_id, _ in sorted_candidates[:3]: # Analisa os top 3
            explanation = training.explain_similarity(most_influential_seed, candidate_id)
            print(f"  - Candidato: {candidate_id}")
            if isinstance(explanation, dict):
                print(f"    - Categorias em comum: {explanation['shared_categories']}")
                print(f"    - Contribuição do Preço: {explanation['price_contribution']}")
                print(f"    - Score Bruto (Dot Product): {explanation['dot_product_score']}")
            else:
                print(f"    - {explanation}")
    else:
        print("Nenhum histórico de usuário para analisar.")

    print("-------------------------------------------------\n")
    
    recommended = [iid for iid, sc in sorted_candidates if iid not in interacted_set][:count]
    if len(recommended) < count:
        for pid in (training._popularity or []):
            if pid not in recommended and pid not in interacted_set:
                recommended.append(pid)
            if len(recommended) >= count: break
    return recommended[:count]