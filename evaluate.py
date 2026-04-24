#!/usr/bin/env python3
"""
evaluate.py — Avaliação e Visualização dos Modelos de Recomendação de Joias

Gera 7 figuras em evaluation/:
  01_data_overview.png       — Exploração dos dados (interações, preço, categorias)
  02_content_based.png       — Matriz de similaridade e distribuição de scores
  03_xgboost.png             — Feature importance, learning curve, predições
  04_kmeans.png              — PCA dos clusters, elbow curve, silhouette
  05_knn.png                 — Distribuição de distâncias entre vizinhos
  06_svd.png                 — Variância explicada, heatmap de ratings
  07_hybrid.png              — Comparação RRF, pesos, decay temporal
"""

import os
import sys
import math
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
PALETTE = sns.color_palette("deep")

OUTPUT_DIR = os.path.join(ROOT, "evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

import joblib

try:
    from utils.constants import MODEL_CACHE_PATH, ITEM_FEATURES_PATH, POPULARITY_PATH
except ImportError:
    MODEL_CACHE_PATH = os.path.join(ROOT, "models")
    ITEM_FEATURES_PATH = os.path.join(MODEL_CACHE_PATH, "item_features.pkl")
    POPULARITY_PATH = os.path.join(MODEL_CACHE_PATH, "popularity.pkl")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_model(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"  [AVISO] Falha ao carregar {os.path.basename(path)}: {e}")
    return None


def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Salvo: {name}")


def try_get_interactions():
    try:
        from db import get_db_connection
        conn = get_db_connection()
        if not conn:
            return None
        df = pd.read_sql_query(
            "SELECT user_id, jewelry_id, type, created_at FROM user_interaction", conn
        )
        conn.close()
        df["type"] = df["type"].astype(str).str.lower()
        df["jewelry_id"] = df["jewelry_id"].astype(str)
        df["user_id"] = df["user_id"].astype(str)
        return df if not df.empty else None
    except Exception:
        return None


def try_get_raw_items():
    try:
        from db import get_db_connection
        conn = get_db_connection()
        if not conn:
            return None
        df = pd.read_sql_query(
            """SELECT id, metal_id, gemstone_id, supplier_id, jewelry_type,
               COALESCE(supplier_price,0)+COALESCE(process_price,0)+COALESCE(profit,0) AS total_price
               FROM jewelries WHERE quantity > 0""",
            conn,
        )
        conn.close()
        return df if not df.empty else None
    except Exception:
        return None


def interaction_score(row):
    TYPE_WEIGHTS = {"ignored": -0.1, "click": 1.0, "shopping_cart": 3.0, "bought": 5.0}
    HALF_LIFE = 30.0
    base = TYPE_WEIGHTS.get(str(row.get("type", "")).lower(), 0.0)
    created = row.get("created_at")
    if created is None:
        days = 0
    else:
        try:
            from datetime import date, datetime
            if isinstance(created, (pd.Timestamp, datetime)):
                created = created.date()
            days = max(0, (date.today() - created).days)
        except Exception:
            days = 0
    lam = math.log(2) / HALF_LIFE
    return base * math.exp(-lam * days)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 1 — Análise Exploratória dos Dados
# ══════════════════════════════════════════════════════════════════════════════

def plot_data_overview(item_df, interactions_df, raw_items_df, popularity):
    print("\n[1/7] Análise Exploratória dos Dados...")

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Análise Exploratória dos Dados", fontsize=16, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # Interações por tipo
    ax = fig.add_subplot(gs[0, 0])
    if interactions_df is not None:
        tc = interactions_df["type"].value_counts()
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(tc))]
        bars = ax.bar(tc.index, tc.values, color=colors, edgecolor="white")
        for bar, v in zip(bars, tc.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:,}", ha="center", va="bottom", fontsize=9)
        ax.set_xlabel("Tipo")
        ax.set_ylabel("Quantidade")
    else:
        ax.text(0.5, 0.5, "Sem dados de interação\n(DB indisponível)",
                ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=11)
    ax.set_title("Distribuição das Interações por Tipo", fontweight="bold")

    # Preço original
    ax = fig.add_subplot(gs[0, 1])
    if raw_items_df is not None:
        prices = raw_items_df["total_price"].clip(0, raw_items_df["total_price"].quantile(0.99))
        ax.hist(prices, bins=40, color=PALETTE[1], edgecolor="white", alpha=0.85)
        ax.axvline(prices.median(), color="red", linestyle="--", lw=1.5,
                   label=f"Mediana: {prices.median():.0f}")
        ax.set_xlabel("Preço Total (R$)")
        ax.set_ylabel("Frequência")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Sem dados (DB indisponível)", ha="center", va="center",
                transform=ax.transAxes, color="gray")
    ax.set_title("Distribuição de Preço (Original)", fontweight="bold")

    # Preço após log1p + StandardScaler
    ax = fig.add_subplot(gs[0, 2])
    if item_df is not None and "total_price" in item_df.columns:
        lp = item_df["total_price"]
        ax.hist(lp, bins=40, color=PALETTE[2], edgecolor="white", alpha=0.85)
        ax.axvline(lp.mean(), color="red", linestyle="--", lw=1.5,
                   label=f"Média: {lp.mean():.2f}")
        ax.set_xlabel("Valor Normalizado")
        ax.set_ylabel("Frequência")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "item_df não disponível", ha="center", va="center",
                transform=ax.transAxes, color="gray")
    ax.set_title("Distribuição de Preço (log1p + StandardScaler)", fontweight="bold")

    # Tipos de joia
    ax = fig.add_subplot(gs[1, 0])
    source = raw_items_df if raw_items_df is not None else None
    if source is not None:
        jt = source["jewelry_type"].value_counts().head(10)
        ax.barh(range(len(jt)), jt.values, color=PALETTE[3], edgecolor="white")
        ax.set_yticks(range(len(jt)))
        ax.set_yticklabels(jt.index, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Quantidade")
    elif item_df is not None:
        jt_cols = [c for c in item_df.columns if c.startswith("jewelry_type_")]
        if jt_cols:
            jt = item_df[jt_cols].sum().sort_values(ascending=False).head(10)
            jt.index = [c.replace("jewelry_type_", "") for c in jt.index]
            ax.barh(range(len(jt)), jt.values, color=PALETTE[3], edgecolor="white")
            ax.set_yticks(range(len(jt)))
            ax.set_yticklabels(jt.index, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("Quantidade")
    ax.set_title("Tipos de Joia (Top 10)", fontweight="bold")

    # Distribuição por metal (inferido do item_df encodado)
    ax = fig.add_subplot(gs[1, 1])
    if item_df is not None:
        metal_cols = [c for c in item_df.columns if c.startswith("metal_id_")]
        if metal_cols:
            ms = item_df[metal_cols].sum().sort_values(ascending=False).head(10)
            ms.index = [c.replace("metal_id_", "Metal ") for c in ms.index]
            ax.barh(range(len(ms)), ms.values, color=PALETTE[4], edgecolor="white")
            ax.set_yticks(range(len(ms)))
            ax.set_yticklabels(ms.index, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("Quantidade de Itens")
        else:
            ax.text(0.5, 0.5, "Colunas metal_id não encontradas",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
    ax.set_title("Distribuição por Metal (Top 10)", fontweight="bold")

    # Top itens por score de popularidade
    ax = fig.add_subplot(gs[1, 2])
    if interactions_df is not None:
        df_pop = interactions_df.copy()
        df_pop["score"] = df_pop.apply(interaction_score, axis=1)
        pop_scores = (
            df_pop.groupby("jewelry_id")["score"].sum()
            .sort_values(ascending=False)
            .head(12)
        )
        short_ids = [str(i)[:8] + "…" for i in pop_scores.index]
        ax.barh(range(len(pop_scores)), pop_scores.values, color=PALETTE[5], edgecolor="white")
        ax.set_yticks(range(len(pop_scores)))
        ax.set_yticklabels(short_ids, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Score de Popularidade")
    elif popularity:
        ax.text(0.5, 0.5, f"{len(popularity)} itens na lista\nde popularidade",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_title("Top 12 Itens por Popularidade", fontweight="bold")

    save_fig(fig, "01_data_overview.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 2 — Content-Based Similarity
# ══════════════════════════════════════════════════════════════════════════════

def plot_content_based(item_df, similarity_matrix, interactions_df):
    print("\n[2/7] Content-Based Similarity...")

    if item_df is None or similarity_matrix is None:
        print("  [SKIP] item_df ou similarity_matrix não disponíveis.")
        return

    n = len(similarity_matrix)
    np.random.seed(42)

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Modelo Content-Based (Similaridade Cosseno por Features)",
                 fontsize=16, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # Distribuição dos scores de similaridade (amostra par-a-par)
    ax = fig.add_subplot(gs[0, 0])
    sample_idx = np.random.choice(n, min(100, n), replace=False)
    sim_sample = similarity_matrix[np.ix_(sample_idx, sample_idx)]
    upper = sim_sample[np.triu_indices(len(sample_idx), k=1)]
    ax.hist(upper, bins=50, color=PALETTE[0], edgecolor="white", alpha=0.85)
    ax.axvline(upper.mean(), color="red", linestyle="--", lw=1.5,
               label=f"Média: {upper.mean():.3f}")
    ax.set_title("Distribuição dos Scores de Similaridade\n(amostra 100×100, par-a-par)",
                 fontweight="bold")
    ax.set_xlabel("Similaridade Cosseno")
    ax.set_ylabel("Frequência")
    ax.legend(fontsize=8)

    # Heatmap de similaridade (amostra 30×30)
    ax = fig.add_subplot(gs[0, 1])
    s = min(30, n)
    idx30 = np.random.choice(n, s, replace=False)
    heat = similarity_matrix[np.ix_(idx30, idx30)]
    cmap = LinearSegmentedColormap.from_list("cb", ["#f7f7f7", "#2166ac"])
    im = ax.imshow(heat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Similaridade")
    ax.set_title(f"Heatmap de Similaridade ({s}×{s} itens aleatórios)", fontweight="bold")
    ax.set_xlabel("Item (índice amostrado)")
    ax.set_ylabel("Item (índice amostrado)")
    ax.set_xticks([])
    ax.set_yticks([])

    # Composição do espaço de features (pizza)
    ax = fig.add_subplot(gs[0, 2])
    feat_cols = [c for c in item_df.columns if c != "id"]
    groups = {}
    prefixes = [("metal_id", "Metal"), ("gemstone_id", "Pedra"),
                ("supplier_id", "Fornecedor"), ("jewelry_type", "Tipo Joia"),
                ("total_price", "Preço")]
    for col in feat_cols:
        matched = False
        for prefix, label in prefixes:
            if col.startswith(prefix):
                groups[label] = groups.get(label, 0) + 1
                matched = True
                break
        if not matched:
            groups["Outros"] = groups.get("Outros", 0) + 1
    sizes = list(groups.values())
    labels_pie = [f"{k}\n({v})" for k, v in groups.items()]
    wedges, _, autotexts = ax.pie(sizes, autopct="%1.0f%%", colors=PALETTE[:len(sizes)],
                                   startangle=90, pctdistance=0.75)
    ax.legend(wedges, labels_pie, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.set_title(f"Composição do Espaço de Features\n({len(feat_cols)} features total)",
                 fontweight="bold")

    # Similaridade máxima por item (melhor vizinho)
    ax = fig.add_subplot(gs[1, 0])
    sim_no_diag = similarity_matrix.copy()
    np.fill_diagonal(sim_no_diag, 0)
    max_sims = np.max(sim_no_diag, axis=1)
    ax.hist(max_sims, bins=40, color=PALETTE[1], edgecolor="white", alpha=0.85)
    ax.axvline(max_sims.mean(), color="red", linestyle="--", lw=1.5,
               label=f"Média: {max_sims.mean():.3f}")
    ax.set_title("Similaridade Máxima por Item\n(melhor vizinho no catálogo)", fontweight="bold")
    ax.set_xlabel("Similaridade Cosseno Máxima")
    ax.set_ylabel("Número de Itens")
    ax.legend(fontsize=8)

    # Scores de interação por tipo (com decay)
    ax = fig.add_subplot(gs[1, 1])
    if interactions_df is not None:
        df_sc = interactions_df.copy()
        df_sc["score"] = df_sc.apply(interaction_score, axis=1)
        for itype, grp in df_sc.groupby("type"):
            vals = grp["score"][grp["score"] > 0]
            if not vals.empty:
                ax.hist(vals, bins=30, alpha=0.65, label=itype)
        ax.set_title("Distribuição dos Scores de Interação\n(por tipo, com decay temporal)",
                     fontweight="bold")
        ax.set_xlabel("Score (base × decay)")
        ax.set_ylabel("Frequência")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Dados de interação não disponíveis",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title("Scores de Interação por Tipo", fontweight="bold")

    # Número de itens similares por threshold
    ax = fig.add_subplot(gs[1, 2])
    thresholds = np.arange(0.05, 1.01, 0.05)
    # Usa linha mediana da matriz como representativa
    ref = similarity_matrix[n // 2]
    counts_thresh = [int(np.sum(ref > t)) for t in thresholds]
    ax.plot(thresholds, counts_thresh, color=PALETTE[2], marker="o", markersize=4, lw=2)
    ax.fill_between(thresholds, counts_thresh, alpha=0.15, color=PALETTE[2])
    ax.set_title("Itens Similares por Threshold\n(referência: item central do catálogo)",
                 fontweight="bold")
    ax.set_xlabel("Threshold de Similaridade")
    ax.set_ylabel("Número de Itens Similares")
    ax.grid(True, alpha=0.4)

    save_fig(fig, "02_content_based.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 3 — XGBoost
# ══════════════════════════════════════════════════════════════════════════════

def plot_xgboost(item_df):
    print("\n[3/7] XGBoost...")

    xgb_model = load_model(os.path.join(MODEL_CACHE_PATH, "xgb_model.pkl"))
    if xgb_model is None or item_df is None:
        print("  [SKIP] Modelo XGBoost ou item_df não disponíveis.")
        return

    feat_cols = [c for c in item_df.columns if c != "id"]
    X_catalog = item_df[feat_cols].values.astype(float)

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Modelo XGBoost (Regressão de Relevância Item-Usuário)",
                 fontsize=16, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # Feature importance (top 20) — ocupa 2 colunas
    ax = fig.add_subplot(gs[0, 0:2])
    try:
        imp = pd.Series(xgb_model.feature_importances_, index=feat_cols)
        imp = imp.sort_values(ascending=False).head(20)
        colors_imp = [PALETTE[0]] * len(imp)
        ax.barh(range(len(imp)), imp.values, color=colors_imp, edgecolor="white")
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(imp.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Importância (gain)")
        ax.set_title("Top 20 Features Mais Importantes (XGBoost — gain)", fontweight="bold")
    except Exception as e:
        ax.text(0.5, 0.5, f"Erro: {e}", ha="center", va="center", transform=ax.transAxes)

    # Distribuição das predições sobre o catálogo
    ax = fig.add_subplot(gs[0, 2])
    try:
        preds = xgb_model.predict(X_catalog)
        ax.hist(preds, bins=40, color=PALETTE[1], edgecolor="white", alpha=0.85)
        ax.axvline(preds.mean(), color="red", linestyle="--", lw=1.5,
                   label=f"Média: {preds.mean():.3f}")
        ax.axvline(np.median(preds), color="orange", linestyle=":", lw=1.5,
                   label=f"Mediana: {np.median(preds):.3f}")
        ax.set_title("Distribuição das Predições XGBoost\n(catálogo completo)", fontweight="bold")
        ax.set_xlabel("Score Predito")
        ax.set_ylabel("Número de Itens")
        ax.legend(fontsize=8)
    except Exception as e:
        ax.text(0.5, 0.5, f"Erro: {e}", ha="center", va="center", transform=ax.transAxes)

    # Curva de aprendizado (eval history) — ocupa 2 colunas
    ax = fig.add_subplot(gs[1, 0:2])
    try:
        evals = xgb_model.evals_result()
        if evals:
            val_key = list(evals.keys())[0]
            metric_key = list(evals[val_key].keys())[0]
            history = evals[val_key][metric_key]
            ax.plot(history, color=PALETTE[0], lw=2, label=f"Validação ({metric_key.upper()})")
            best = int(np.argmin(history))
            ax.axvline(best, color="red", linestyle="--", lw=1.5,
                       label=f"Best iter: {best}  |  {metric_key.upper()}={history[best]:.4f}")
            ax.set_title(f"Curva de Aprendizado — XGBoost (Early Stopping)\n"
                         f"Total de árvores treinadas: {len(history)}",
                         fontweight="bold")
            ax.set_xlabel("Iteração (número de árvores)")
            ax.set_ylabel(metric_key.upper())
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.35)
        else:
            ax.text(0.5, 0.5, "Histórico de treino não disponível\nno modelo salvo.",
                    ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=11)
    except Exception as e:
        ax.text(0.5, 0.5, f"Histórico não disponível\n({e})",
                ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=10)
    ax.set_title("Curva de Aprendizado — XGBoost", fontweight="bold")

    # Box plot de predições por quartil de preço
    ax = fig.add_subplot(gs[1, 2])
    try:
        preds = xgb_model.predict(X_catalog)
        price = item_df["total_price"].values
        quartis = np.percentile(price, [0, 25, 50, 75, 100])
        groups = []
        for i in range(4):
            mask = (price >= quartis[i]) & (price <= quartis[i + 1])
            groups.append(preds[mask])
        ax.boxplot(groups, labels=["Q1\n(baixo)", "Q2", "Q3", "Q4\n(alto)"],
                   patch_artist=True,
                   boxprops=dict(facecolor=PALETTE[2], alpha=0.7),
                   medianprops=dict(color="red", lw=2))
        ax.set_title("Score XGBoost por Faixa de Preço\n(quartis do preço normalizado)",
                     fontweight="bold")
        ax.set_xlabel("Faixa de Preço")
        ax.set_ylabel("Score Predito")
    except Exception as e:
        ax.text(0.5, 0.5, f"Erro: {e}", ha="center", va="center", transform=ax.transAxes)

    save_fig(fig, "03_xgboost.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 4 — K-Means
# ══════════════════════════════════════════════════════════════════════════════

def plot_kmeans(item_df):
    print("\n[4/7] K-Means (inclui elbow + silhouette — pode levar alguns segundos)...")

    kmeans_data = load_model(os.path.join(MODEL_CACHE_PATH, "kmeans_model.pkl"))
    if kmeans_data is None or item_df is None:
        print("  [SKIP] Modelo K-Means ou item_df não disponíveis.")
        return

    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, silhouette_samples

    feat_cols = [c for c in item_df.columns if c != "id"]
    X = item_df[feat_cols].values.astype(float)
    labels = kmeans_data["labels"]
    model = kmeans_data["model"]
    n_clusters = model.n_clusters
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Modelo K-Means (Clusterização de Joias)",
                 fontsize=16, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # PCA 2D — ocupa 2 colunas
    ax = fig.add_subplot(gs[0, 0:2])
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    for k in range(n_clusters):
        mask = labels == k
        ax.scatter(X2[mask, 0], X2[mask, 1], c=[cluster_colors[k]],
                   s=25, alpha=0.7, label=f"Cluster {k}")
    centroids2 = pca.transform(model.cluster_centers_)
    ax.scatter(centroids2[:, 0], centroids2[:, 1], c="black", s=200,
               marker="X", zorder=6, label="Centróides")
    var = pca.explained_variance_ratio_
    ax.set_title(f"Visualização PCA dos Clusters K-Means\n"
                 f"PC1={var[0]:.1%} | PC2={var[1]:.1%} da variância total",
                 fontweight="bold")
    ax.set_xlabel(f"PC1 ({var[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var[1]:.1%})")
    ax.legend(loc="upper right", fontsize=7, ncol=2)

    # Tamanho dos clusters
    ax = fig.add_subplot(gs[0, 2])
    sizes = pd.Series(labels).value_counts().sort_index()
    bar_colors = [cluster_colors[i] for i in sizes.index]
    bars = ax.bar(sizes.index, sizes.values, color=bar_colors, edgecolor="white")
    for bar, val in zip(bars, sizes.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                str(val), ha="center", va="bottom", fontsize=8)
    ax.axhline(sizes.mean(), color="red", linestyle="--", lw=1.5,
               label=f"Média: {sizes.mean():.0f} itens")
    ax.set_title("Tamanho dos Clusters", fontweight="bold")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Número de Itens")
    ax.legend(fontsize=8)

    # Elbow curve
    ax = fig.add_subplot(gs[1, 0])
    k_max = min(16, len(X))
    k_range = range(2, k_max)
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        km.fit(X)
        inertias.append(km.inertia_)
    ax.plot(list(k_range), inertias, color=PALETTE[0], marker="o", ms=5, lw=2)
    ax.axvline(n_clusters, color="red", linestyle="--", lw=1.5,
               label=f"k atual = {n_clusters}")
    ax.set_title("Elbow Curve — Inércia vs. Número de Clusters", fontweight="bold")
    ax.set_xlabel("Número de Clusters (k)")
    ax.set_ylabel("Inércia (WCSS)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    # Silhouette score por k
    ax = fig.add_subplot(gs[1, 1])
    sil_scores = []
    sil_range = range(2, k_max)
    for k in sil_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        km_labels = km.fit_predict(X)
        sil = silhouette_score(X, km_labels, sample_size=min(500, len(X)), random_state=42)
        sil_scores.append(sil)
    best_k = list(sil_range)[int(np.argmax(sil_scores))]
    ax.plot(list(sil_range), sil_scores, color=PALETTE[1], marker="o", ms=5, lw=2)
    ax.axvline(n_clusters, color="red", linestyle="--", lw=1.5, label=f"k atual = {n_clusters}")
    ax.axvline(best_k, color="green", linestyle=":", lw=1.5,
               label=f"Melhor k = {best_k}  ({max(sil_scores):.3f})")
    ax.set_title("Silhouette Score por k\n(−1=ruim | 0=sobreposição | +1=ótimo)",
                 fontweight="bold")
    ax.set_xlabel("Número de Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    # Silhouette plot (diagrama de faixas) para k atual
    ax = fig.add_subplot(gs[1, 2])
    sil_vals = silhouette_samples(X, labels)
    y_lower = 10
    for k in range(n_clusters):
        k_vals = np.sort(sil_vals[labels == k])
        y_upper = y_lower + len(k_vals)
        color = plt.cm.nipy_spectral(float(k) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, k_vals, alpha=0.75, color=color)
        ax.text(-0.07, y_lower + len(k_vals) * 0.5, str(k), fontsize=7)
        y_lower = y_upper + 5
    avg_sil = float(np.mean(sil_vals))
    ax.axvline(avg_sil, color="red", linestyle="--", lw=1.5, label=f"Média: {avg_sil:.3f}")
    ax.set_title(f"Silhouette por Cluster (k={n_clusters})\n(cada faixa = 1 cluster)",
                 fontweight="bold")
    ax.set_xlabel("Silhouette Score")
    ax.set_ylabel("Amostras (agrupadas por cluster)")
    ax.legend(fontsize=8)

    save_fig(fig, "04_kmeans.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 5 — KNN
# ══════════════════════════════════════════════════════════════════════════════

def plot_knn(item_df):
    print("\n[5/7] KNN...")

    knn_data = load_model(os.path.join(MODEL_CACHE_PATH, "knn_model.pkl"))
    if knn_data is None or item_df is None:
        print("  [SKIP] Modelo KNN ou item_df não disponíveis.")
        return

    model = knn_data["model"]
    X = knn_data["X"]
    metric = knn_data.get("metric", "cosine")

    print("  Calculando distâncias (kneighbors)...")
    distances, _ = model.kneighbors(X)
    neighbor_dists = distances[:, 1:]  # exclui o próprio item (dist=0)
    k_actual = neighbor_dists.shape[1]

    fig = plt.figure(figsize=(18, 7))
    fig.suptitle(f"Modelo KNN (K-Nearest Neighbors | métrica: {metric} | k={k_actual})",
                 fontsize=16, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    # Distribuição de todas as distâncias
    ax = fig.add_subplot(gs[0])
    all_dists = neighbor_dists.flatten()
    ax.hist(all_dists, bins=55, color=PALETTE[0], edgecolor="white", alpha=0.85)
    ax.axvline(all_dists.mean(), color="red", linestyle="--", lw=1.5,
               label=f"Média: {all_dists.mean():.4f}")
    ax.axvline(np.median(all_dists), color="orange", linestyle=":", lw=1.5,
               label=f"Mediana: {np.median(all_dists):.4f}")
    ax.set_title(f"Distribuição das Distâncias ({metric})\nEntre Itens e seus {k_actual} Vizinhos",
                 fontweight="bold")
    ax.set_xlabel(f"Distância ({metric})")
    ax.set_ylabel("Frequência")
    ax.legend(fontsize=8)

    # Distância média por posição do vizinho
    ax = fig.add_subplot(gs[1])
    mean_per_k = np.mean(neighbor_dists, axis=0)
    std_per_k = np.std(neighbor_dists, axis=0)
    kpos = np.arange(1, k_actual + 1)
    ax.plot(kpos, mean_per_k, color=PALETTE[1], marker="o", ms=4, lw=2, label="Média")
    ax.fill_between(kpos, mean_per_k - std_per_k, mean_per_k + std_per_k,
                    alpha=0.2, color=PALETTE[1], label="±1 DP")
    ax.set_title("Distância Média por Posição do Vizinho\n(k=1 = vizinho mais próximo)",
                 fontweight="bold")
    ax.set_xlabel("Posição do Vizinho (k)")
    ax.set_ylabel(f"Distância ({metric})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    # Box plot de distâncias em posições selecionadas
    ax = fig.add_subplot(gs[2])
    positions = sorted({0, k_actual // 4, k_actual // 2, 3 * k_actual // 4, k_actual - 1})
    positions = [p for p in positions if 0 <= p < k_actual]
    data_box = [neighbor_dists[:, p] for p in positions]
    ax.boxplot(data_box, labels=[f"k={p+1}" for p in positions],
               patch_artist=True,
               boxprops=dict(facecolor=PALETTE[2], alpha=0.7),
               medianprops=dict(color="red", lw=2))
    ax.set_title("Dispersão das Distâncias\npor Posição do Vizinho", fontweight="bold")
    ax.set_xlabel("Posição do Vizinho")
    ax.set_ylabel(f"Distância ({metric})")

    save_fig(fig, "05_knn.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 6 — SVD Collaborative Filtering
# ══════════════════════════════════════════════════════════════════════════════

def plot_svd(interactions_df):
    print("\n[6/7] SVD Collaborative Filtering...")

    svd_data = load_model(os.path.join(MODEL_CACHE_PATH, "svd_model.pkl"))
    if svd_data is None:
        print("  [SKIP] Modelo SVD não disponível.")
        return

    ratings = svd_data["ratings"]
    n_users, n_items = ratings.shape

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Modelo SVD — Filtragem Colaborativa (Matrix Factorization)",
                 fontsize=16, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # Variância explicada por fator latente — ocupa 2 colunas
    ax = fig.add_subplot(gs[0, 0:2])
    if interactions_df is not None:
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import svds

            df_svd = interactions_df.copy()
            df_svd["score"] = df_svd.apply(interaction_score, axis=1)
            user_codes = df_svd["user_id"].astype("category")
            item_codes = df_svd["jewelry_id"].astype("category")
            pivot = csr_matrix(
                (df_svd["score"], (user_codes.cat.codes, item_codes.cat.codes))
            )
            k = min(20, pivot.shape[1] - 1)
            _, sigma_vals, _ = svds(pivot.asfptype(), k=k)
            sigma_vals = sigma_vals[::-1]  # svds retorna em ordem crescente

            total_var = float(pivot.power(2).sum())
            var_per_comp = (sigma_vals ** 2) / total_var
            cumulative = np.cumsum(var_per_comp)

            x_pos = np.arange(1, k + 1)
            ax.bar(x_pos, var_per_comp * 100, color=PALETTE[0], edgecolor="white",
                   alpha=0.85, label="Variância por componente")
            ax_twin = ax.twinx()
            ax_twin.plot(x_pos, cumulative * 100, color="red", marker="o", ms=4,
                         lw=2, label="Acumulada")
            ax_twin.axhline(80, color="gray", linestyle=":", lw=1, label="80%")
            ax_twin.set_ylabel("Variância Acumulada (%)", color="red")
            ax_twin.tick_params(axis="y", colors="red")
            ax_twin.set_ylim(0, 105)
            ax.set_xticks(x_pos)
            lines1, lab1 = ax.get_legend_handles_labels()
            lines2, lab2 = ax_twin.get_legend_handles_labels()
            ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="upper right")
            ax.set_title(f"Variância Explicada por Fator Latente SVD\n"
                         f"({k} fatores | variância total capturada: {cumulative[-1]:.1%})",
                         fontweight="bold")
            ax.set_xlabel("Componente Latente")
            ax.set_ylabel("Variância Explicada (%)")
        except Exception as e:
            ax.text(0.5, 0.5, f"Erro ao executar SVD: {e}",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
    else:
        ax.text(0.5, 0.5, "Dados de interação não disponíveis\npara análise de variância",
                ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=11)
    ax.set_title("Variância Explicada por Fator Latente SVD", fontweight="bold")

    # Dimensões da matriz
    ax = fig.add_subplot(gs[0, 2])
    labels_dim = ["Usuários\n(linhas)", "Itens\n(colunas)"]
    vals_dim = [n_users, n_items]
    bars = ax.bar(labels_dim, vals_dim,
                  color=[PALETTE[0], PALETTE[1]], edgecolor="white", width=0.5)
    for bar, val in zip(bars, vals_dim):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                f"{val:,}", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_title(f"Dimensões da Matriz User-Item\n(Fatoração SVD)", fontweight="bold")
    ax.set_ylabel("Contagem")

    # Heatmap da matriz reconstruída (amostra) — ocupa 2 colunas
    ax = fig.add_subplot(gs[1, 0:2])
    np.random.seed(42)
    su = min(20, n_users)
    si = min(30, n_items)
    ui = np.random.choice(n_users, su, replace=False)
    ii = np.random.choice(n_items, si, replace=False)
    sample_mtx = ratings[np.ix_(ui, ii)]
    im = ax.imshow(sample_mtx, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="Score Previsto")
    ax.set_title(f"Heatmap da Matriz de Ratings Reconstruída\n({su} usuários × {si} itens — amostra aleatória)",
                 fontweight="bold")
    ax.set_xlabel("Item (amostra)")
    ax.set_ylabel("Usuário (amostra)")
    ax.set_xticks([])
    ax.set_yticks([])

    # Distribuição dos scores preditos
    ax = fig.add_subplot(gs[1, 2])
    flat = ratings.flatten()
    sample_flat = flat[np.random.choice(len(flat), min(15000, len(flat)), replace=False)]
    ax.hist(sample_flat, bins=50, color=PALETTE[3], edgecolor="white", alpha=0.85)
    ax.axvline(sample_flat.mean(), color="red", linestyle="--", lw=1.5,
               label=f"Média: {sample_flat.mean():.3f}")
    ax.set_title("Distribuição dos Scores Preditos pelo SVD\n(amostra da matriz reconstruída)",
                 fontweight="bold")
    ax.set_xlabel("Score Previsto")
    ax.set_ylabel("Frequência")
    ax.legend(fontsize=8)

    save_fig(fig, "06_svd.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 7 — Comparação dos Algoritmos e Sistema Híbrido
# ══════════════════════════════════════════════════════════════════════════════

def plot_hybrid(item_df):
    print("\n[7/7] Sistema Híbrido (RRF)...")

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        "Sistema Híbrido de Recomendação — Reciprocal Rank Fusion (RRF)",
        fontsize=16, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    ALGOS = ["Content-Based\n(CB)", "KNN", "SVD\n(Colaborativo)", "K-Means"]
    WEIGHTS = [1.0, 0.9, 0.8, 0.6]
    ALGO_COLORS = PALETTE[:4]

    # Pesos dos algoritmos
    ax = fig.add_subplot(gs[0, 0])
    bars = ax.bar(ALGOS, WEIGHTS, color=ALGO_COLORS, edgecolor="white", width=0.6)
    for bar, w in zip(bars, WEIGHTS):
        ax.text(bar.get_x() + bar.get_width() / 2, w + 0.01,
                f"{w:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_title("Pesos dos Algoritmos no RRF", fontweight="bold")
    ax.set_ylabel("Peso")
    ax.set_ylim(0, 1.25)
    ax.axhline(1.0, color="gray", linestyle=":", lw=1)

    # Score RRF por posição para cada algoritmo
    ax = fig.add_subplot(gs[0, 1])
    positions = np.arange(1, 21)
    for name, w, color in zip(ALGOS, WEIGHTS, ALGO_COLORS):
        ax.plot(positions, (1.0 / positions) * w, marker="o", ms=3, lw=2,
                label=name.replace("\n", " "), color=color)
    ax.set_title("Score RRF por Posição na Lista\n[score = (1/pos) × peso]",
                 fontweight="bold")
    ax.set_xlabel("Posição na Lista de Recomendações")
    ax.set_ylabel("Score RRF")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    # Status dos arquivos de modelo
    ax = fig.add_subplot(gs[0, 2])
    MODEL_FILES = {
        "item_features.pkl": "Features dos Itens",
        "similarity_matrix.pkl": "Similaridade (CB)",
        "xgb_model.pkl": "XGBoost",
        "kmeans_model.pkl": "K-Means",
        "knn_model.pkl": "KNN",
        "svd_model.pkl": "SVD Colaborativo",
        "popularity.pkl": "Popularidade (fallback)",
    }
    f_labels, f_sizes, f_colors = [], [], []
    for fname, label in MODEL_FILES.items():
        path = os.path.join(MODEL_CACHE_PATH, fname)
        size = os.path.getsize(path) / 1024 if os.path.exists(path) else 0
        f_labels.append(label)
        f_sizes.append(max(size, 0.1))
        f_colors.append("#2ecc71" if size > 0 else "#e74c3c")
    bars2 = ax.barh(range(len(f_labels)), f_sizes, color=f_colors, edgecolor="white")
    for i, (bar, size) in enumerate(zip(bars2, f_sizes)):
        text = f"{size:.0f} KB" if size > 0.1 else "AUSENTE"
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                text, va="center", fontsize=8)
    ax.set_yticks(range(len(f_labels)))
    ax.set_yticklabels(f_labels, fontsize=8)
    ax.set_title("Status dos Modelos Salvos\n(verde=disponível | vermelho=ausente)",
                 fontweight="bold")
    ax.set_xlabel("Tamanho (KB)")

    # Simulação de contribuição RRF — ocupa 2 colunas
    ax = fig.add_subplot(gs[1, 0:2])
    np.random.seed(0)
    N = 15
    all_ranks = [np.random.permutation(N) for _ in WEIGHTS]
    rrf_total = np.zeros(N)
    for ranks, w in zip(all_ranks, WEIGHTS):
        for item_i, rank in enumerate(ranks):
            rrf_total[item_i] += (1.0 / (rank + 1)) * w

    order = np.argsort(rrf_total)[::-1]
    x = np.arange(N)
    width = 0.18
    for i, (ranks, w, name, color) in enumerate(zip(all_ranks, WEIGHTS, ALGOS, ALGO_COLORS)):
        contrib = np.array([(1.0 / (ranks[j] + 1)) * w for j in range(N)])
        ax.bar(x + i * width, contrib[order], width, label=name.replace("\n", " "),
               color=color, edgecolor="white", alpha=0.85)
    ax.plot(x + 1.5 * width, rrf_total[order], "k--o", ms=4, lw=1.5,
            label="Score RRF Total", zorder=5)
    ax.set_title(
        "Simulação: Contribuição de Cada Algoritmo para o Score RRF Final\n"
        "(rankings aleatórios | itens ordenados pelo score total)",
        fontweight="bold",
    )
    ax.set_xlabel("Item (ordem decrescente de score RRF)")
    ax.set_ylabel("Score RRF por Algoritmo")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"#{i+1}" for i in range(N)], fontsize=7)
    ax.legend(fontsize=8, loc="upper right")

    # Decay temporal dos scores de interação
    ax = fig.add_subplot(gs[1, 2])
    days_range = np.linspace(0, 120, 300)
    HALF_LIFE = 30.0
    lam = math.log(2) / HALF_LIFE
    TYPE_W = {"ignored": -0.1, "click": 1.0, "shopping_cart": 3.0, "bought": 5.0}
    for (itype, base), color in zip(TYPE_W.items(), PALETTE[:4]):
        scores = [base * math.exp(-lam * d) for d in days_range]
        ax.plot(days_range, scores, lw=2, label=f"{itype} (base={base})", color=color)
    ax.axhline(0, color="gray", linestyle="--", lw=1, alpha=0.5)
    ax.axvline(HALF_LIFE, color="gray", linestyle=":", lw=1, alpha=0.7,
               label=f"Meia-vida ({HALF_LIFE:.0f}d)")
    ax.set_title("Decay Temporal dos Scores de Interação\n(half_life = 30 dias)",
                 fontweight="bold")
    ax.set_xlabel("Dias desde a interação")
    ax.set_ylabel("Score (base × decay)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    save_fig(fig, "07_hybrid.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  AVALIAÇÃO DOS MODELOS DE RECOMENDAÇÃO DE JOIAS")
    print("=" * 62)
    print(f"  Saída: {OUTPUT_DIR}\n")

    # Carrega modelos do disco
    print("[MODELOS] Carregando artefatos salvos...")
    item_df = load_model(ITEM_FEATURES_PATH)
    sim_matrix = load_model(os.path.join(MODEL_CACHE_PATH, "similarity_matrix.pkl"))
    popularity = load_model(POPULARITY_PATH)

    status = lambda v, extra="": f"OK ({extra})" if v is not None else "AUSENTE"
    print(f"  item_df          : {status(item_df, f'{len(item_df)} itens' if item_df is not None else '')}")
    print(f"  similarity_matrix: {status(sim_matrix, f'{sim_matrix.shape}' if sim_matrix is not None else '')}")
    print(f"  popularity       : {status(popularity, f'{len(popularity)} itens' if popularity is not None else '')}")
    print(f"  xgb_model        : {status(load_model(os.path.join(MODEL_CACHE_PATH, 'xgb_model.pkl')))}")
    print(f"  kmeans_model     : {status(load_model(os.path.join(MODEL_CACHE_PATH, 'kmeans_model.pkl')))}")
    print(f"  knn_model        : {status(load_model(os.path.join(MODEL_CACHE_PATH, 'knn_model.pkl')))}")
    print(f"  svd_model        : {status(load_model(os.path.join(MODEL_CACHE_PATH, 'svd_model.pkl')))}")

    # Tenta banco de dados
    print("\n[BANCO] Buscando dados...")
    interactions_df = try_get_interactions()
    raw_items_df = try_get_raw_items()
    if interactions_df is not None:
        print(f"  interactions: {len(interactions_df)} linhas  |  "
              f"usuários únicos: {interactions_df['user_id'].nunique()}")
    else:
        print("  DB indisponível — gráficos dependentes de interações serão omitidos/simplificados.")
    if raw_items_df is not None:
        print(f"  raw_items   : {len(raw_items_df)} itens")

    # Gera figuras
    plot_data_overview(item_df, interactions_df, raw_items_df, popularity)
    plot_content_based(item_df, sim_matrix, interactions_df)
    plot_xgboost(item_df)
    plot_kmeans(item_df)
    plot_knn(item_df)
    plot_svd(interactions_df)
    plot_hybrid(item_df)

    print("\n" + "=" * 62)
    print(f"  CONCLUÍDO! Figuras salvas em: {OUTPUT_DIR}")
    print("=" * 62)


if __name__ == "__main__":
    main()
