import os
import random
from datetime import datetime, date
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

from db import get_db_connection
from training import training

try:
    from xgboost import XGBRegressor
except Exception as e:
    XGBRegressor = None
    print("[xgboost_training] Aviso: xgboost não está instalado. Instale com: pip install xgboost")

try:
    from utils.constants import MODEL_CACHE_PATH, XGB_MODEL_PATH
except Exception:
    MODEL_CACHE_PATH = os.path.join("model_cache")
    XGB_MODEL_PATH = os.path.join(MODEL_CACHE_PATH, "xgb_model.pkl")

try:
    from utils.constants import MODEL_CACHE_PATH, XGB_MODEL_PATH
except Exception:
    MODEL_CACHE_PATH = os.path.join("model_cache")
    XGB_MODEL_PATH = os.path.join(MODEL_CACHE_PATH, "xgb_model.pkl")

os.makedirs(MODEL_CACHE_PATH, exist_ok=True)

XGB_MODEL_FILE = XGB_MODEL_PATH

def _fetch_interactions(limit_days: Optional[int] = None) -> pd.DataFrame:
    conn = get_db_connection()
    if conn is None:
        raise RuntimeError("Não foi possível conectar ao banco para buscar interações.")
    try:
        q = "SELECT user_id, jewelry_id, type, created_at FROM user_interaction"
        params = None
        if limit_days is not None:
            q += " WHERE created_at >= current_date - %s"
            params = (limit_days,)
        with conn.cursor() as cur:
            if params:
                cur.execute(q, params)
            else:
                cur.execute(q)
            rows = cur.fetchall()
        if not rows:
            return pd.DataFrame(columns=["user_id", "jewelry_id", "type", "created_at"])
        df = pd.DataFrame(rows, columns=["user_id", "jewelry_id", "type", "created_at"])
        df["type"] = df["type"].astype(str).str.lower()
        df["jewelry_id"] = df["jewelry_id"].astype(str)
        df["user_id"] = df["user_id"].astype(str)
        return df
    finally:
        conn.close()

def _build_item_feature_matrix() -> pd.DataFrame:
    training._load_cache_if_needed()
    if training._item_df is None:
        df = training.load_items_from_db()
        if df is None or df.empty:
            raise RuntimeError("Nenhum item disponível para treinar o XGBoost.")
        return df
    return training._item_df

def _prepare_supervised_dataset(limit_days: Optional[int] = 365, neg_ratio: int = 3) -> Tuple[pd.DataFrame, pd.Series]:
    print("[xgboost_training] Preparando dataset supervisionado...")
    interactions = _fetch_interactions(limit_days=limit_days)
    item_df = _build_item_feature_matrix()
    if interactions.empty:
        raise RuntimeError("Não há interações para construir dataset supervisionado.")
    item_features = item_df.set_index('id')
    all_item_ids = set(item_features.index.tolist())

    X_rows = []
    y = []

    for _, row in interactions.iterrows():
        jid = str(row["jewelry_id"])
        if jid not in item_features.index:
            continue
        try:
            label = float(training.interaction_score(row))
        except Exception:
            label = 1.0
        feat = item_features.loc[jid].copy()
        if 'id' in feat.index:
            feat = feat.drop('id')
        X_rows.append(feat.values.astype(float))
        y.append(label)

        interacted_by_user = set(interactions[interactions["user_id"] == row["user_id"]]["jewelry_id"].astype(str).tolist())
        candidates = list(all_item_ids - interacted_by_user)
        if not candidates:
            continue
        for _ in range(neg_ratio):
            neg_jid = random.choice(candidates)
            neg_feat = item_features.loc[neg_jid].copy()
            if 'id' in neg_feat.index:
                neg_feat = neg_feat.drop('id')
            X_rows.append(neg_feat.values.astype(float))
            y.append(0.0)

    if not X_rows:
        raise RuntimeError("Nenhuma linha foi gerada para treino (X vazio).")

    X = np.vstack(X_rows)
    y = np.array(y, dtype=float)

    feature_cols = [c for c in item_features.columns if c != 'id']
    if len(feature_cols) != X.shape[1]:
        feature_cols = [f"f{i}" for i in range(X.shape[1])]
    Xdf = pd.DataFrame(X, columns=feature_cols)
    yser = pd.Series(y, name="target")
    print(f"[xgboost_training] Dataset pronto. Shape X: {Xdf.shape}, y: {yser.shape}, positive_ratio: {np.mean(yser>0):.3f}")
    return Xdf, yser

def train_xgb(limit_days: Optional[int] = 365,
                neg_ratio: int = 3,
                test_size: float = 0.2,
                random_state: int = 42,
                xgb_params: Optional[dict] = None,
                persist: bool = True) -> dict:
    if XGBRegressor is None:
        raise RuntimeError("xgboost não encontrado. Instale com: pip install xgboost")

    X, y = _prepare_supervised_dataset(limit_days=limit_days, neg_ratio=neg_ratio)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    default_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": random_state,
        "verbosity": 0,
        "n_jobs": -1,
        "eval_metric": "rmse",
        "early_stopping_rounds": 25
    }
    if xgb_params:
        default_params.update(xgb_params)

    model = XGBRegressor(**default_params)
    print("[xgboost_training] Iniciando fit do XGBoost...")
    #model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=25, verbose=False)
    #model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="rmse", verbose=False)
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_test, y_test)], 
        verbose=False
    )

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)

    result = {
        "rmse": float(rmse),
        "mae": float(mae),
        "model": model
    }

    if persist:
        joblib.dump(model, XGB_MODEL_FILE)
        print(f"[xgboost_training] Modelo salvo em: {XGB_MODEL_FILE}")

    print(f"[xgboost_training] Treino concluído. RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return result

def load_xgb_model() -> Optional[XGBRegressor]:
    if not os.path.exists(XGB_MODEL_FILE):
        print(f"[xgboost_training] Modelo XGB não encontrado em {XGB_MODEL_FILE}")
        return None
    try:
        model = joblib.load(XGB_MODEL_FILE)
        print(f"[xgboost_training] Modelo carregado de: {XGB_MODEL_FILE}")
        return model
    except Exception as e:
        print(f"[xgboost_training] Erro ao carregar modelo XGB: {e}")
        return None

def recommend_with_xgb(user_id: str, count: int = 10) -> List[str]:
    training._load_cache_if_needed()
    if training._item_df is None:
        print("[xgboost_training] Features de item não carregadas. Retornando vazio.") # 🟢 NOVO LOG
        return []

    model = load_xgb_model()
    if model is None:
        print("[xgboost_training] Modelo XGB indisponível — usando fallback de popularidade.")
        return (training._popularity or [])[:count]

    conn = get_db_connection()
    interacted_set = set()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT jewelry_id FROM user_interaction WHERE user_id = %s;", (user_id,))
            rows = cur.fetchall()
            if rows:
                interacted_set = {str(r[0]) for r in rows}
    finally:
        conn.close()

    item_df = training._item_df.copy()
    cand_df = item_df[~item_df['id'].isin(interacted_set)].reset_index(drop=True)
    
    if cand_df.empty:
        print(f"[xgboost_training] Usuário {user_id} não tem candidatos não-interagidos (total interações: {len(interacted_set)}).")
        return (training._popularity or [])[:count] 

    feat_cols = [c for c in cand_df.columns if c != 'id']
    X_cand = cand_df[feat_cols].values.astype(float)

    preds = model.predict(X_cand)
    cand_df["pred_score"] = preds
    cand_df["id_str"] = cand_df["id"].astype(str)

    top = cand_df.sort_values("pred_score", ascending=False).head(count)["id_str"].tolist()

    if len(top) < count:
        for pid in (training._popularity or []):
            if pid not in top and pid not in interacted_set:
                top.append(pid)
            if len(top) >= count:
                break

    return top[:count]

if __name__ == "__main__":
    print("=== xgboost_training.py ===")
    print("1) Treinar modelo XGBoost com dados (limit_days=365, neg_ratio=3)...")
    try:
        metrics = train_xgb(limit_days=365, neg_ratio=3)
        print("Treino OK. Métricas:", {k: v for k, v in metrics.items() if k in ("rmse", "mae")})
    except Exception as e:
        print("Erro durante treino:", e)