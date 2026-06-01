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
    HALF_LIFE = 120.0
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
    model = kmeans_data["model"]
    labels = model.predict(X)  # recomputa com os dados atuais
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
            k = min(30, pivot.shape[1] - 1)
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

    try:
        from utils.constants import (RRF_WEIGHT_CB, RRF_WEIGHT_KNN,
                                     RRF_WEIGHT_SVD, RRF_WEIGHT_KMEANS,
                                     HALF_LIFE_DAYS, TYPE_WEIGHTS)
    except ImportError:
        RRF_WEIGHT_CB = RRF_WEIGHT_KNN = RRF_WEIGHT_SVD = RRF_WEIGHT_KMEANS = 1.0
        HALF_LIFE_DAYS = 120.0
        TYPE_WEIGHTS = {"ignored": -0.1, "click": 1.0, "shopping_cart": 3.0, "bought": 5.0}

    ALGOS   = ["Content-Based\n(CB)", "KNN", "SVD\n(Colaborativo)", "K-Means"]
    WEIGHTS = [RRF_WEIGHT_CB, RRF_WEIGHT_KNN, RRF_WEIGHT_SVD, RRF_WEIGHT_KMEANS]
    ALGO_COLORS = PALETTE[:4]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Sistema Híbrido de Recomendação — Reciprocal Rank Fusion (RRF)",
        fontsize=16, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

    # ── Linha 0: pesos atuais / score por posição / status dos modelos ────────

    # Pesos atuais (lidos de constants.py)
    ax = fig.add_subplot(gs[0, 0])
    w_max = max(max(WEIGHTS), 1.0)
    bars = ax.bar(ALGOS, WEIGHTS, color=ALGO_COLORS, edgecolor="white", width=0.6)
    for bar, w in zip(bars, WEIGHTS):
        ax.text(bar.get_x() + bar.get_width() / 2, w + w_max * 0.02,
                f"{w:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_title("Pesos Atuais dos Algoritmos\n(constants.py → RRF_WEIGHT_*)",
                 fontweight="bold")
    ax.set_ylabel("Peso RRF")
    ax.set_ylim(0, w_max * 1.25)
    ax.axhline(1.0, color="gray", linestyle=":", lw=1, alpha=0.6)

    # Score RRF por posição (com os pesos atuais)
    ax = fig.add_subplot(gs[0, 1])
    positions = np.arange(1, 21)
    for name, w, color in zip(ALGOS, WEIGHTS, ALGO_COLORS):
        ax.plot(positions, (1.0 / positions) * w, marker="o", ms=3, lw=2,
                label=f"{name.replace(chr(10),' ')} (w={w:.2f})", color=color)
    ax.set_title("Score RRF por Posição\n[score = (1/pos) × peso]", fontweight="bold")
    ax.set_xlabel("Posição na lista do algoritmo")
    ax.set_ylabel("Contribuição ao score final")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.4)

    # Status dos arquivos de modelo
    ax = fig.add_subplot(gs[0, 2])
    MODEL_FILES = {
        "item_features.pkl":    "Features dos Itens",
        "similarity_matrix.pkl":"Similaridade (CB)",
        "xgb_model.pkl":        "XGBoost",
        "kmeans_model.pkl":     "K-Means",
        "knn_model.pkl":        "KNN",
        "svd_model.pkl":        "SVD Colaborativo",
        "popularity.pkl":       "Popularidade (fallback)",
    }
    f_labels, f_sizes, f_colors_bar = [], [], []
    for fname, label in MODEL_FILES.items():
        path = os.path.join(MODEL_CACHE_PATH, fname)
        size = os.path.getsize(path) / 1024 if os.path.exists(path) else 0
        f_labels.append(label)
        f_sizes.append(max(size, 0.1))
        f_colors_bar.append("#2ecc71" if size > 0 else "#e74c3c")
    bars2 = ax.barh(range(len(f_labels)), f_sizes, color=f_colors_bar, edgecolor="white")
    for bar, size in zip(bars2, f_sizes):
        text = f"{size:.0f} KB" if size > 0.1 else "AUSENTE"
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                text, va="center", fontsize=8)
    ax.set_yticks(range(len(f_labels)))
    ax.set_yticklabels(f_labels, fontsize=8)
    ax.set_title("Status dos Modelos Salvos\n(verde=OK | vermelho=ausente)", fontweight="bold")
    ax.set_xlabel("Tamanho (KB)")

    # ── Linha 1: sensibilidade dos pesos ──────────────────────────────────────

    weight_range = np.linspace(0.0, 2.0, 80)
    # Posições representativas na lista do algoritmo
    POSITIONS = [1, 3, 5, 10, 20]
    POOL = 20   # tamanho do pool (count*2)

    for col_idx, (algo_name, base_w, color) in enumerate(zip(ALGOS[:3], WEIGHTS[:3], ALGO_COLORS[:3])):
        ax = fig.add_subplot(gs[1, col_idx])
        for pos in POSITIONS:
            rrf_vals = [(1.0 / pos) * w for w in weight_range]
            ax.plot(weight_range, rrf_vals, lw=1.8,
                    label=f"posição {pos}",
                    linestyle="-" if pos <= 5 else "--")
        ax.axvline(base_w, color="red", linestyle="--", lw=1.5,
                   label=f"atual ({base_w:.2f})")
        ax.set_title(f"Sensibilidade: {algo_name.replace(chr(10), ' ')}\n"
                     f"score = (1/posição) × peso", fontweight="bold")
        ax.set_xlabel("Peso do algoritmo")
        ax.set_ylabel("Contribuição ao score RRF")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.35)

    # ── Linha 2: dominância relativa e simulação de contribuição ──────────────

    # Dominância: para cada peso variado isoladamente, % de itens que um algoritmo
    # coloca no top-10 final (simulação com rankings aleatórios, 500 runs)
    ax = fig.add_subplot(gs[2, 0])
    np.random.seed(42)
    N_RUNS = 300
    TOP_N  = 10
    w_steps = np.linspace(0.1, 2.0, 20)

    for algo_idx, (algo_name, base_w, color) in enumerate(zip(ALGOS, WEIGHTS, ALGO_COLORS)):
        dom_vals = []
        for test_w in w_steps:
            trial_weights = list(WEIGHTS)
            trial_weights[algo_idx] = test_w
            count_in_top = 0
            for _ in range(N_RUNS):
                all_ranks = [np.random.permutation(POOL) for _ in range(4)]
                item_scores = np.zeros(POOL)
                for r, w in zip(all_ranks, trial_weights):
                    for item_i, rank in enumerate(r):
                        item_scores[item_i] += (1.0 / (rank + 1)) * w
                top_items = set(np.argsort(item_scores)[::-1][:TOP_N])
                algo_items = set(np.where(all_ranks[algo_idx] < TOP_N)[0])
                count_in_top += len(top_items & algo_items)
            dom_vals.append(count_in_top / (N_RUNS * TOP_N))
        ax.plot(w_steps, dom_vals, lw=2, color=color,
                label=algo_name.replace("\n", " "), marker="o", ms=3)
    for algo_idx, (base_w, color) in enumerate(zip(WEIGHTS, ALGO_COLORS)):
        ax.axvline(base_w, color=color, linestyle=":", lw=1, alpha=0.5)
    ax.set_title(f"Dominância Simulada: % top-{TOP_N} ocupado\nao variar peso de cada algoritmo",
                 fontweight="bold")
    ax.set_xlabel("Peso do algoritmo variado")
    ax.set_ylabel(f"Fração média do top-{TOP_N}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.35)
    ax.set_ylim(0, 1)

    # Contribuição RRF com pesos atuais (simulação fixa)
    ax = fig.add_subplot(gs[2, 1])
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
    ax.set_title("Simulação: Contribuição por Algoritmo\n(rankings aleatórios, pesos atuais)",
                 fontweight="bold")
    ax.set_xlabel("Item (ranking final decrescente)")
    ax.set_ylabel("Score RRF")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"#{i+1}" for i in range(N)], fontsize=7)
    ax.legend(fontsize=7, loc="upper right")

    # Decay temporal
    ax = fig.add_subplot(gs[2, 2])
    days_range = np.linspace(0, 120, 300)
    lam = math.log(2) / HALF_LIFE_DAYS
    for (itype, base), color in zip(TYPE_WEIGHTS.items(), PALETTE[:4]):
        scores_decay = [base * math.exp(-lam * d) for d in days_range]
        ax.plot(days_range, scores_decay, lw=2, label=f"{itype} (base={base})", color=color)
    ax.axhline(0, color="gray", linestyle="--", lw=1, alpha=0.5)
    ax.axvline(HALF_LIFE_DAYS, color="gray", linestyle=":", lw=1.2, alpha=0.7,
               label=f"Meia-vida ({HALF_LIFE_DAYS:.0f}d)")
    ax.set_title(f"Decay Temporal dos Scores\n(HALF_LIFE_DAYS = {HALF_LIFE_DAYS:.0f})",
                 fontweight="bold")
    ax.set_xlabel("Dias desde a interação")
    ax.set_ylabel("Score (base × decay)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    save_fig(fig, "07_hybrid.png")


# ══════════════════════════════════════════════════════════════════════════════
# TABELA 1 — Estatísticas do Dataset
# ══════════════════════════════════════════════════════════════════════════════

PAPER_DIR = os.path.join(ROOT, "paper_output")


def _split_train_val_test(interactions_df):
    """
    Aplica filtro de usuários com ≥ 3 interações e divide por tempo:
      - teste     : última interação de cada usuário elegível
      - validação : penúltima interação de cada usuário elegível
      - treino    : demais interações dos usuários elegíveis

    Retorna (train_df, val_df, test_df, eligible_users).
    """
    counts = interactions_df.groupby("user_id")["jewelry_id"].count()
    eligible = counts[counts >= 3].index
    df = interactions_df[interactions_df["user_id"].isin(eligible)].copy()
    df = df.sort_values(["user_id", "created_at"])

    test_idx = df.groupby("user_id").tail(1).index
    remaining = df.drop(index=test_idx)
    val_idx = remaining.groupby("user_id").tail(1).index

    test_df  = df.loc[test_idx]
    val_df   = remaining.loc[val_idx]
    train_df = remaining.drop(index=val_idx)

    return train_df, val_df, test_df, eligible


def _build_rankers(item_df, train_eval_df, sim_matrix, cb_threshold=None, knn_data_override=None, svd_n_factors=20):
    """
    Retorna dict {nome_modelo: rank_fn(uid) -> list[item_id]}.
    train_eval_df é o histórico visível do modelo (treino ou treino+val).
    """
    id_list   = item_df["id"].tolist()
    id_to_idx = {iid: i for i, iid in enumerate(id_list)}

    def seed_scores(uid):
        ss = {}
        for _, r in train_eval_df[train_eval_df["user_id"] == uid].iterrows():
            iid = str(r["jewelry_id"])
            ss[iid] = ss.get(iid, 0.0) + interaction_score(r)
        return ss

    def rank_cb(uid):
        ss = seed_scores(uid)
        interacted = set(ss)
        cand = {}
        for sid, w in ss.items():
            if sid not in id_to_idx:
                continue
            sidx = id_to_idx[sid]
            for idx, sv in enumerate(sim_matrix[sidx]):
                if cb_threshold is not None and sv < cb_threshold:
                    continue
                iid = id_list[idx]
                if iid in interacted:
                    continue
                cand[iid] = cand.get(iid, 0.0) + sv * w
        return sorted(cand, key=cand.get, reverse=True)

    knn_data = knn_data_override if knn_data_override is not None else load_model(os.path.join(MODEL_CACHE_PATH, "knn_model.pkl"))

    def rank_knn(uid):
        if knn_data is None:
            return []
        ss = seed_scores(uid)
        interacted = set(ss)
        knn_ids  = knn_data["item_ids"]
        knn_idx  = {iid: i for i, iid in enumerate(knn_ids)}
        X_knn    = knn_data["X"]
        model_knn = knn_data["model"]
        cand = {}
        for sid, w in ss.items():
            if sid not in knn_idx:
                continue
            vec = X_knn[knn_idx[sid]].reshape(1, -1)
            dists, indices = model_knn.kneighbors(vec)
            for dist, ni in zip(dists[0], indices[0]):
                nb = knn_ids[ni]
                if nb in interacted:
                    continue
                cand[nb] = cand.get(nb, 0.0) + (1.0 - dist) * w
        return sorted(cand, key=cand.get, reverse=True)

    km_data = load_model(os.path.join(MODEL_CACHE_PATH, "kmeans_model.pkl"))

    def rank_kmeans(uid):
        if km_data is None:
            return []
        ss = seed_scores(uid)
        interacted = set(ss)
        id_to_cl = dict(zip(km_data["item_ids"], km_data["labels"]))
        cw = {}
        for sid, w in ss.items():
            cl = id_to_cl.get(sid)
            if cl is not None:
                cw[cl] = cw.get(cl, 0.0) + w
        cand = {iid: cw[id_to_cl[iid]]
                for iid in km_data["item_ids"]
                if iid not in interacted and id_to_cl.get(iid) in cw}
        return sorted(cand, key=cand.get, reverse=True)

    # Re-treina SVD apenas com train_eval_df para evitar data leakage
    _svd_eval_data = None
    if not train_eval_df.empty:
        try:
            from scipy.sparse import csr_matrix as _csr
            from scipy.sparse.linalg import svds as _svds

            _df = train_eval_df.copy()
            _df["score"] = _df.apply(interaction_score, axis=1)
            _df["user_id"] = _df["user_id"].astype(str)
            _df["jewelry_id"] = _df["jewelry_id"].astype(str)
            _uc = _df["user_id"].astype("category")
            _ic = _df["jewelry_id"].astype("category")
            _pivot = _csr((_df["score"], (_uc.cat.codes, _ic.cat.codes)))
            _k = max(1, min(svd_n_factors, _pivot.shape[0] - 1, _pivot.shape[1] - 1))
            _u, _s, _vt = _svds(_pivot.asfptype(), k=_k)
            _ratings = np.dot(np.dot(_u, np.diag(_s)), _vt)
            _svd_eval_data = {
                "ratings":  _ratings,
                "user_map": _uc.cat.categories,
                "item_map": _ic.cat.categories,
            }
        except Exception as _e:
            print(f"  [AVISO] SVD eval falhou: {_e}")

    def rank_svd(uid):
        if _svd_eval_data is None:
            return []
        user_map = list(_svd_eval_data["user_map"])
        if uid not in user_map:
            return []
        interacted = set(seed_scores(uid))
        u_idx   = user_map.index(uid)
        ratings = _svd_eval_data["ratings"][u_idx]
        return [str(_svd_eval_data["item_map"][i])
                for i in np.argsort(ratings)[::-1]
                if str(_svd_eval_data["item_map"][i]) not in interacted]

    xgb_model = load_model(os.path.join(MODEL_CACHE_PATH, "xgb_model.pkl"))

    def rank_xgb(uid):
        if xgb_model is None:
            return []
        interacted = set(seed_scores(uid))
        feat_cols = [c for c in item_df.columns if c != "id"]
        preds  = xgb_model.predict(item_df[feat_cols].values.astype(float))
        ranked = [id_list[i] for i in np.argsort(preds)[::-1]]
        return [iid for iid in ranked if iid not in interacted]

    try:
        from utils.constants import (RRF_WEIGHT_CB, RRF_WEIGHT_KNN,
                                     RRF_WEIGHT_SVD, RRF_WEIGHT_KMEANS)
    except ImportError:
        RRF_WEIGHT_CB = RRF_WEIGHT_KNN = RRF_WEIGHT_SVD = RRF_WEIGHT_KMEANS = 1.0

    def rank_rrf(uid):
        rankings = [
            (rank_cb(uid),     RRF_WEIGHT_CB),
            (rank_knn(uid),    RRF_WEIGHT_KNN),
            (rank_svd(uid),    RRF_WEIGHT_SVD),
            (rank_kmeans(uid), RRF_WEIGHT_KMEANS),
        ]
        scores = {}
        for ranked, w in rankings:
            for i, iid in enumerate(ranked):
                scores[iid] = scores.get(iid, 0.0) + w / (60 + i + 1)
        return sorted(scores, key=scores.get, reverse=True)

    return {
        "CB (Content-Based)": rank_cb,
        "KNN":                 rank_knn,
        "K-Means":             rank_kmeans,
        "SVD (Colaborativo)":  rank_svd,
        "XGBoost":             rank_xgb,
        "Hibrido (RRF)":       rank_rrf,
    }


def _eval_metrics(rankers, eval_df, k=10):
    """
    Computa Precision@k, Recall@k e NDCG@k para cada modelo em eval_df.
    Assume 1 item relevante por usuário (leave-one-out).
    Retorna dict {modelo: {"precision": float, "recall": float, "ndcg": float}}.
    """
    results = {name: {"precision": [], "recall": [], "ndcg": []} for name in rankers}

    for uid, grp in eval_df.groupby("user_id"):
        rel = str(grp.iloc[0]["jewelry_id"])
        for name, rank_fn in rankers.items():
            top_k = rank_fn(uid)[:k]
            hit   = rel in top_k
            rank  = (top_k.index(rel) + 1) if hit else None
            results[name]["precision"].append(1 / k if hit else 0.0)
            results[name]["recall"].append(1.0 if hit else 0.0)
            results[name]["ndcg"].append(1.0 / np.log2(rank + 1) if hit else 0.0)

    return {
        name: {m: float(np.mean(v)) for m, v in metrics.items()}
        for name, metrics in results.items()
    }


def generate_model_comparison(item_df, interactions_df, sim_matrix, cb_threshold=None, knn_data_override=None, svd_n_factors=20):
    """Avalia todos os modelos no conjunto de teste e gera tab3_model_comparison.csv."""
    print("\n[TAB3] Gerando comparação de modelos (conjunto de teste)...")
    os.makedirs(PAPER_DIR, exist_ok=True)

    if interactions_df is None or item_df is None or sim_matrix is None:
        print("  [SKIP] Dados insuficientes.")
        return None

    train_df, val_df, test_df, _ = _split_train_val_test(interactions_df)
    if test_df.empty:
        print("  [SKIP] Conjunto de teste vazio.")
        return None

    # Modelo vê treino + validação (tudo exceto a última interação)
    train_eval_df = pd.concat([train_df, val_df], ignore_index=True)
    rankers = _build_rankers(item_df, train_eval_df, sim_matrix, cb_threshold=cb_threshold, knn_data_override=knn_data_override, svd_n_factors=svd_n_factors)
    metrics = _eval_metrics(rankers, test_df)

    rows = [
        (name,
         f"{m['precision']:.4f}",
         f"{m['recall']:.4f}",
         f"{m['ndcg']:.4f}")
        for name, m in metrics.items()
    ]
    tab3 = pd.DataFrame(rows, columns=["Modelo", "Precision@10", "Recall@10", "NDCG@10"])
    tab3.to_csv(os.path.join(PAPER_DIR, "tab3_model_comparison.csv"), index=False)
    print(f"  -> tab3_model_comparison.csv")
    for _, r in tab3.iterrows():
        print(f"     {r['Modelo']:25s}  P@10={r['Precision@10']}  R@10={r['Recall@10']}  NDCG@10={r['NDCG@10']}")

    return metrics


def plot_model_comparison(metrics):
    """Gera gráfico de barras agrupadas (Fig. 1) em paper_output/fig1_model_comparison.png."""
    if metrics is None:
        print("  [SKIP] Métricas não disponíveis para Fig. 1.")
        return

    print("\n[FIG1] Gerando gráfico de comparação de modelos...")
    os.makedirs(PAPER_DIR, exist_ok=True)

    model_names = list(metrics.keys())
    precision   = [metrics[m]["precision"] for m in model_names]
    recall      = [metrics[m]["recall"]    for m in model_names]
    ndcg        = [metrics[m]["ndcg"]      for m in model_names]

    x     = np.arange(len(model_names))
    width = 0.25
    colors = [PALETTE[0], PALETTE[1], PALETTE[2]]

    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - width, precision, width, label="Precision@10", color=colors[0], edgecolor="white")
    b2 = ax.bar(x,          recall,   width, label="Recall@10",    color=colors[1], edgecolor="white")
    b3 = ax.bar(x + width,  ndcg,     width, label="NDCG@10",      color=colors[2], edgecolor="white")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Valor da métrica")
    ax.set_xlabel("Modelo")
    ax.set_title("Comparação dos Modelos de Recomendação (conjunto de teste)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    path = os.path.join(PAPER_DIR, "fig1_model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> fig1_model_comparison.png")


def generate_hybrid_gain_table(metrics):
    """Computa ganho percentual do Hibrido (RRF) vs. cada modelo isolado e salva tab4_hybrid_gain.csv."""
    print("\n[TAB4] Gerando tabela de ganho da arquitetura hibrida...")
    os.makedirs(PAPER_DIR, exist_ok=True)

    if metrics is None:
        print("  [SKIP] Metricas nao disponiveis.")
        return

    hybrid = metrics.get("Hibrido (RRF)")
    if hybrid is None:
        print("  [SKIP] Metricas do modelo hibrido nao encontradas.")
        return

    components = ["CB (Content-Based)", "KNN", "K-Means", "SVD (Colaborativo)"]
    metric_keys = [("precision", "Precision@10"), ("recall", "Recall@10"), ("ndcg", "NDCG@10")]

    def delta(hybrid_val, base_val):
        if base_val == 0:
            return "+inf" if hybrid_val > 0 else "0,00%"
        return f"{((hybrid_val - base_val) / base_val) * 100:+.2f}%"

    rows = []
    for model in components:
        base = metrics.get(model)
        if base is None:
            rows.append((model, "N/A", "N/A", "N/A"))
            continue
        rows.append((
            model,
            delta(hybrid["precision"], base["precision"]),
            delta(hybrid["recall"],    base["recall"]),
            delta(hybrid["ndcg"],      base["ndcg"]),
        ))

    tab4 = pd.DataFrame(rows, columns=[
        "Modelo comparado", "Delta Precision@10 (%)", "Delta Recall@10 (%)", "Delta NDCG@10 (%)"
    ])
    tab4.to_csv(os.path.join(PAPER_DIR, "tab4_hybrid_gain.csv"), index=False)
    print(f"  -> tab4_hybrid_gain.csv")
    for _, r in tab4.iterrows():
        print(f"     Hibrido vs {r['Modelo comparado']:25s}  "
              f"P={r['Delta Precision@10 (%)']:>10s}  "
              f"R={r['Delta Recall@10 (%)']:>10s}  "
              f"NDCG={r['Delta NDCG@10 (%)']:>10s}")



def plot_k_sensitivity(item_df, interactions_df, sim_matrix, cb_threshold=None, knn_data_override=None, svd_n_factors=20):
    """Fig. 3 — Precision@K e Recall@K do modelo hibrido para K de 1 a 20."""
    print("\n[FIG3] Sensibilidade ao parametro K (hibrido RRF)...")
    os.makedirs(PAPER_DIR, exist_ok=True)

    if interactions_df is None or item_df is None or sim_matrix is None:
        print("  [SKIP] Dados insuficientes.")
        return

    train_df, val_df, test_df, _ = _split_train_val_test(interactions_df)
    if test_df.empty:
        print("  [SKIP] Conjunto de teste vazio.")
        return

    train_eval = pd.concat([train_df, val_df], ignore_index=True)
    rankers    = _build_rankers(item_df, train_eval, sim_matrix, cb_threshold=cb_threshold, knn_data_override=knn_data_override, svd_n_factors=svd_n_factors)
    rank_rrf   = rankers["Hibrido (RRF)"]

    # Pre-computa rankings para cada usuario de teste
    user_rankings = {}
    for uid, grp in test_df.groupby("user_id"):
        rel = str(grp.iloc[0]["jewelry_id"])
        user_rankings[uid] = (rank_rrf(uid), rel)

    ks = list(range(1, 21))
    precisions, recalls = [], []
    for k in ks:
        p_vals, r_vals = [], []
        for uid, (ranked, rel) in user_rankings.items():
            hit = rel in ranked[:k]
            p_vals.append((1 / k) if hit else 0.0)
            r_vals.append(1.0   if hit else 0.0)
        precisions.append(float(np.mean(p_vals)) if p_vals else 0.0)
        recalls.append(float(np.mean(r_vals))    if r_vals else 0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ks, precisions, marker="o", lw=2, color=PALETTE[0], label="Precision@K")
    ax.plot(ks, recalls,    marker="s", lw=2, color=PALETTE[1], label="Recall@K")
    ax.axvline(10, color="gray", linestyle="--", lw=1.5, label="K = 10 (escolhido)")
    ax.set_xlabel("K (tamanho da lista de recomendacao)")
    ax.set_ylabel("Valor da metrica")
    ax.set_title("Sensibilidade das Metricas ao Parametro K\n(Arquitetura Hibrida RRF — conjunto de teste)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xticks(ks)
    ax.set_xlim(1, 20)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    fig.savefig(os.path.join(PAPER_DIR, "fig3_k_sensitivity.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig3_k_sensitivity.png")


def plot_kmeans_clusters(item_df, interactions_df, sim_matrix, cb_threshold=None, knn_data_override=None, svd_n_factors=20):
    """Fig. 4 — PCA 2D do espaco de features com clusters K-Means e usuario-exemplo."""
    from sklearn.decomposition import PCA as _PCA

    print("\n[FIG4] Visualizacao 2D dos clusters K-Means...")
    os.makedirs(PAPER_DIR, exist_ok=True)

    if item_df is None:
        print("  [SKIP] item_df nao disponivel.")
        return

    km_data = load_model(os.path.join(MODEL_CACHE_PATH, "kmeans_model.pkl"))
    if km_data is None:
        print("  [SKIP] Modelo K-Means nao disponivel.")
        return

    feat_cols = [c for c in item_df.columns if c != "id"]
    X  = item_df[feat_cols].values.astype(float)
    ids = item_df["id"].tolist()

    pca = _PCA(n_components=2, random_state=42)
    X2  = pca.fit_transform(X)
    id_to_xy = {iid: X2[i] for i, iid in enumerate(ids)}

    id_to_cl = dict(zip(km_data["item_ids"], km_data["labels"]))
    labels   = np.array([id_to_cl.get(iid, -1) for iid in ids])
    n_cl     = km_data["model"].n_clusters
    colors   = plt.cm.tab10(np.linspace(0, 1, n_cl))

    fig, ax = plt.subplots(figsize=(13, 8))

    for k in range(n_cl):
        mask = labels == k
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   c=[colors[k]], s=15, alpha=0.45, label=f"Cluster {k}")

    if interactions_df is not None and not interactions_df.empty:
        train_df, val_df, test_df, eligible = _split_train_val_test(interactions_df)
        if len(eligible) > 0:
            sample_uid  = str(eligible[0])
            train_eval  = pd.concat([train_df, val_df], ignore_index=True)
            history_ids = (train_eval[train_eval["user_id"] == sample_uid]
                           ["jewelry_id"].astype(str).tolist())
            rankers = _build_rankers(item_df, train_eval, sim_matrix, cb_threshold=cb_threshold, knn_data_override=knn_data_override, svd_n_factors=svd_n_factors)
            recs    = rankers["Hibrido (RRF)"](sample_uid)[:10]

            h_pts = np.array([id_to_xy[i] for i in history_ids if i in id_to_xy])
            r_pts = np.array([id_to_xy[i] for i in recs        if i in id_to_xy])

            if len(h_pts):
                ax.scatter(h_pts[:, 0], h_pts[:, 1], c="black", s=140,
                           marker="*", zorder=6, linewidths=0.5,
                           label=f"Historico do usuario ({len(h_pts)} itens)")
            if len(r_pts):
                ax.scatter(r_pts[:, 0], r_pts[:, 1], c="red", s=100,
                           marker="D", zorder=5, linewidths=0.5,
                           label="Recomendados (top 10)")

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]:.1%} da variancia)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} da variancia)")
    ax.set_title("Espaco de Features 2D — Clusters K-Means com Historico e Recomendacoes",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, ncol=2)

    fig.savefig(os.path.join(PAPER_DIR, "fig4_kmeans_clusters.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig4_kmeans_clusters.png")


def plot_cold_start(item_df, interactions_df, sim_matrix, cb_threshold=None, knn_data_override=None, svd_n_factors=20):
    """Fig. 5 — Precision@10 media por modelo em funcao do tamanho do historico."""
    print("\n[FIG5] Comportamento dos modelos sob cold start...")
    os.makedirs(PAPER_DIR, exist_ok=True)

    if interactions_df is None or item_df is None or sim_matrix is None:
        print("  [SKIP] Dados insuficientes.")
        return

    history_sizes = [1, 2, 3, 5, 10, 20]
    model_names   = ["CB (Content-Based)", "KNN", "K-Means",
                     "SVD (Colaborativo)", "XGBoost", "Hibrido (RRF)"]
    results  = {m: [] for m in model_names}
    valid_ks = []

    df_sorted = interactions_df.sort_values(["user_id", "created_at"])

    for n in history_sizes:
        train_rows, test_rows = [], []
        for uid, grp in df_sorted.groupby("user_id"):
            if len(grp) < n + 1:
                continue
            # n interacoes antes da ultima
            train_rows.append(grp.iloc[-(n + 1):-1])
            test_rows.append(grp.iloc[[-1]])

        if not train_rows:
            for m in model_names:
                results[m].append(np.nan)
            valid_ks.append(n)
            continue

        train_n = pd.concat(train_rows, ignore_index=True)
        test_n  = pd.concat(test_rows,  ignore_index=True)

        rankers = _build_rankers(item_df, train_n, sim_matrix, cb_threshold=cb_threshold, knn_data_override=knn_data_override, svd_n_factors=svd_n_factors)
        metrics = _eval_metrics(rankers, test_n)

        for m in model_names:
            results[m].append(metrics.get(m, {}).get("precision", np.nan))
        valid_ks.append(n)

    fig, ax = plt.subplots(figsize=(11, 6))
    markers = ["o", "s", "^", "D", "v", "*"]
    for i, m in enumerate(model_names):
        vals = results[m]
        ax.plot(valid_ks, vals, marker=markers[i], lw=2,
                color=PALETTE[i % len(PALETTE)], label=m, markersize=6, alpha=0.85)

    ax.set_xlabel("Numero de interacoes no historico do usuario")
    ax.set_ylabel("Precision@10 media")
    ax.set_title("Comportamento dos Modelos sob Cold Start\n(Precision@10 x tamanho do historico)",
                 fontweight="bold")
    ax.set_xticks(valid_ks)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    fig.savefig(os.path.join(PAPER_DIR, "fig5_cold_start.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig5_cold_start.png")


def generate_dataset_table(raw_items_df, interactions_df):
    """Gera tab1_dataset.csv e as três tabelas de distribuição em paper_output/."""
    print("\n[TAB1] Gerando tabela de estatísticas do dataset...")
    os.makedirs(PAPER_DIR, exist_ok=True)

    rows = []

    # ── Catálogo ──────────────────────────────────────────────────────────────
    n_items = len(raw_items_df) if raw_items_df is not None else "N/A"
    rows.append(("Número total de joias no catálogo", n_items))

    # ── Usuários / interações ─────────────────────────────────────────────────
    if interactions_df is not None:
        n_users        = interactions_df["user_id"].nunique()
        n_interactions = len(interactions_df)
        n_items_ui     = interactions_df["jewelry_id"].nunique()
        density        = n_interactions / (n_users * (n_items if n_items != "N/A" else n_items_ui))
        date_min       = interactions_df["created_at"].min()
        date_max       = interactions_df["created_at"].max()
        if hasattr(date_min, "date"):
            date_min = date_min.date()
        if hasattr(date_max, "date"):
            date_max = date_max.date()
    else:
        n_users = n_interactions = density = date_min = date_max = "N/A"

    rows.append(("Número total de usuários", n_users))
    rows.append(("Número total de interações", n_interactions))
    rows.append(("Densidade da matriz usuário-item (%)",
                 f"{density * 100:.4f}%" if density != "N/A" else "N/A"))
    rows.append(("Data inicial do histórico", date_min))
    rows.append(("Data final do histórico",   date_max))

    # ── Distribuição por categoria ────────────────────────────────────────────
    if raw_items_df is not None:
        jt_counts = raw_items_df["jewelry_type"].value_counts().sort_index()
        mt_counts = raw_items_df["metal_id"].value_counts().sort_index()
        gm_counts = raw_items_df["gemstone_id"].value_counts().sort_index()

        rows.append(("Tipos de joia distintos (jewelry_type)", len(jt_counts)))
        rows.append(("Metais distintos (metal_id)",            len(mt_counts)))
        rows.append(("Pedras distintas (gemstone_id)",         len(gm_counts)))

        # Distribuições individuais
        jt_counts.rename_axis("jewelry_type").reset_index(name="Quantidade").to_csv(
            os.path.join(PAPER_DIR, "tab1_dist_jewelry_type.csv"), index=False
        )
        mt_counts.rename_axis("metal_id").reset_index(name="Quantidade").to_csv(
            os.path.join(PAPER_DIR, "tab1_dist_metal_id.csv"), index=False
        )
        gm_counts.rename_axis("gemstone_id").reset_index(name="Quantidade").to_csv(
            os.path.join(PAPER_DIR, "tab1_dist_gemstone_id.csv"), index=False
        )
        print("  -> tab1_dist_jewelry_type.csv, tab1_dist_metal_id.csv, tab1_dist_gemstone_id.csv")
    else:
        rows.append(("Tipos de joia distintos (jewelry_type)", "N/A"))
        rows.append(("Metais distintos (metal_id)",            "N/A"))
        rows.append(("Pedras distintas (gemstone_id)",         "N/A"))

    # ── Divisão treino / validação / teste ────────────────────────────────────
    if interactions_df is not None:
        train_df, val_df, test_df, eligible = _split_train_val_test(interactions_df)
        rows.append(("Usuários elegíveis para avaliação (≥ 3 interações)", len(eligible)))
        rows.append(("Interações no conjunto de treino",                    len(train_df)))
        rows.append(("Interações no conjunto de validação",                 len(val_df)))
        rows.append(("Interações no conjunto de teste",                     len(test_df)))
        rows.append(("Usuários no conjunto de validação",                   val_df["user_id"].nunique()))
        rows.append(("Usuários no conjunto de teste",                       test_df["user_id"].nunique()))
    else:
        for label in (
            "Usuários elegíveis para avaliação (≥ 3 interações)",
            "Interações no conjunto de treino",
            "Interações no conjunto de validação",
            "Interações no conjunto de teste",
            "Usuários no conjunto de validação",
            "Usuários no conjunto de teste",
        ):
            rows.append((label, "N/A"))

    # ── Salva CSV principal ───────────────────────────────────────────────────
    tab1 = pd.DataFrame(rows, columns=["Indicador", "Valor"])
    out_path = os.path.join(PAPER_DIR, "tab1_dataset.csv")
    tab1.to_csv(out_path, index=False)
    print(f"  -> tab1_dataset.csv  ({len(tab1)} linhas)")


# ══════════════════════════════════════════════════════════════════════════════
# TABELA 2 — Hiperparâmetros finais por modelo
# ══════════════════════════════════════════════════════════════════════════════

def generate_hyperparams_table(item_df, interactions_df, sim_matrix):
    """Computa NDCG@10 na validação para cada modelo e gera tab2_hyperparams.csv."""
    print("\n[TAB2] Gerando tabela de hiperparâmetros...")
    os.makedirs(PAPER_DIR, exist_ok=True)

    if interactions_df is None or item_df is None or sim_matrix is None:
        print("  [SKIP] item_df, interactions_df ou sim_matrix ausentes.")
        return None

    try:
        from utils.constants import (KNN_N_NEIGHBORS, KMEANS_N_CLUSTERS,
                                     RRF_WEIGHT_CB, RRF_WEIGHT_KNN,
                                     RRF_WEIGHT_SVD, RRF_WEIGHT_KMEANS,
                                     CB_SIMILARITY_THRESHOLD, SVD_N_FACTORS,
                                     GRID_SEARCH_CB, GRID_SEARCH_KNN, GRID_SEARCH_SVD)
    except ImportError:
        KNN_N_NEIGHBORS   = 20
        KMEANS_N_CLUSTERS = 10
        RRF_WEIGHT_CB = RRF_WEIGHT_KNN = RRF_WEIGHT_SVD = RRF_WEIGHT_KMEANS = 1.0
        CB_SIMILARITY_THRESHOLD = 0.1
        SVD_N_FACTORS = 20
        GRID_SEARCH_CB = GRID_SEARCH_KNN = GRID_SEARCH_SVD = True

    train_df, val_df, _, _ = _split_train_val_test(interactions_df)
    if val_df.empty:
        print("  [SKIP] Conjunto de validação vazio.")
        return None

    # CB: grid search ou valor fixo de constants.py
    if GRID_SEARCH_CB:
        cb_grid = [None, 0.1, 0.2, 0.3, 0.5]
        cb_results = {}
        for t in cb_grid:
            r = _build_rankers(item_df, train_df, sim_matrix, cb_threshold=t)
            m = _eval_metrics(r, val_df)
            cb_results[t] = m["CB (Content-Based)"]["ndcg"]
        cb_best = max(cb_results, key=cb_results.get)
        cb_ndcg = cb_results[cb_best]
    else:
        cb_best = CB_SIMILARITY_THRESHOLD
        r = _build_rankers(item_df, train_df, sim_matrix, cb_threshold=cb_best)
        cb_ndcg = _eval_metrics(r, val_df)["CB (Content-Based)"]["ndcg"]
        print(f"  [CB] Grid search desabilitado — usando threshold={cb_best}")
    cb_best_str = "Sem corte" if cb_best is None else str(cb_best).replace(".", ",")

    # KNN: grid search ou valor fixo de constants.py
    from sklearn.neighbors import NearestNeighbors as _NearestNeighbors
    feat_cols_knn = [c for c in item_df.columns if c != "id"]
    X_knn = item_df[feat_cols_knn].values.astype(float)
    if GRID_SEARCH_KNN:
        knn_grid = [5, 10, 15, 20, 30]
        knn_results = {}
        print("  Grid search KNN: ", end="", flush=True)
        for k in knn_grid:
            n_neighbors = min(k + 1, len(X_knn))
            tmp_model = _NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
            tmp_model.fit(X_knn)
            tmp_knn_data = {
                "model": tmp_model,
                "item_ids": item_df["id"].tolist(),
                "X": X_knn,
                "metric": "cosine",
            }
            r = _build_rankers(item_df, train_df, sim_matrix, cb_threshold=cb_best, knn_data_override=tmp_knn_data)
            m = _eval_metrics(r, val_df)
            knn_results[k] = (m["KNN"]["ndcg"], tmp_knn_data)
            print(f"K={k}({m['KNN']['ndcg']:.4f}) ", end="", flush=True)
        print()
        knn_best = max(knn_results, key=lambda k: knn_results[k][0])
        knn_ndcg, knn_best_data = knn_results[knn_best]
    else:
        knn_best = KNN_N_NEIGHBORS
        knn_best_data = None  # _build_rankers carrega o modelo do disco
        r = _build_rankers(item_df, train_df, sim_matrix, cb_threshold=cb_best)
        knn_ndcg = _eval_metrics(r, val_df)["KNN"]["ndcg"]
        print(f"  [KNN] Grid search desabilitado — usando K={knn_best}")

    # SVD: grid search ou valor fixo de constants.py
    if GRID_SEARCH_SVD:
        svd_grid = [5, 10, 15, 20, 25, 30]
        svd_results = {}
        print("  Grid search SVD: ", end="", flush=True)
        for k in svd_grid:
            r = _build_rankers(item_df, train_df, sim_matrix, cb_threshold=cb_best,
                               knn_data_override=knn_best_data, svd_n_factors=k)
            m = _eval_metrics(r, val_df)
            svd_results[k] = m["SVD (Colaborativo)"]["ndcg"]
            print(f"k={k}({svd_results[k]:.4f}) ", end="", flush=True)
        print()
        svd_best = max(svd_results, key=svd_results.get)
        svd_ndcg = svd_results[svd_best]
    else:
        svd_best = SVD_N_FACTORS
        r = _build_rankers(item_df, train_df, sim_matrix, cb_threshold=cb_best,
                           knn_data_override=knn_best_data, svd_n_factors=svd_best)
        svd_ndcg = _eval_metrics(r, val_df)["SVD (Colaborativo)"]["ndcg"]
        print(f"  [SVD] Grid search desabilitado — usando k={svd_best} fatores")

    # Avalia todos os modelos com os hiperparâmetros escolhidos
    rankers = _build_rankers(item_df, train_df, sim_matrix, cb_threshold=cb_best,
                             knn_data_override=knn_best_data, svd_n_factors=svd_best)
    metrics = _eval_metrics(rankers, val_df)

    rows_out = [
        ("CB (Content-Based)",
         "Threshold de similaridade {Sem corte; 0,1; 0,2; 0,3; 0,5}",
         f"threshold = {cb_best_str}",
         f"{cb_ndcg:.4f}"),
        ("KNN",
         "K {5, 10, 15, 20, 30}  |  metrica = cosseno",
         f"K = {knn_best}  |  metrica = cosseno",
         f"{knn_ndcg:.4f}"),
        ("K-Means",
         "k {5, 8, 10, 12, 15}  |  n_init = 10",
         f"k = {KMEANS_N_CLUSTERS}  |  n_init = 10",
         f"{metrics['K-Means']['ndcg']:.4f}"),
        ("SVD (Colaborativo)",
         "Fatores latentes {5, 10, 15, 20}",
         f"k = {svd_best} fatores latentes",
         f"{svd_ndcg:.4f}"),
        ("XGBoost",
         "max_depth {3,4,5,6,8}; lr {0,01; 0,05; 0,1}; n_est {100,200,300,500}",
         "max_depth=6 | lr=0,05 | n_est=300 | early_stop=25",
         f"{metrics['XGBoost']['ndcg']:.4f}"),
        ("Hibrido (RRF)",
         "w1,w2,w3,w4 {0,5; 0,8; 1,0; 1,2; 1,5} (busca em grade)",
         f"w_CB={RRF_WEIGHT_CB} | w_KNN={RRF_WEIGHT_KNN} | w_SVD={RRF_WEIGHT_SVD} | w_KM={RRF_WEIGHT_KMEANS}",
         f"{metrics['Hibrido (RRF)']['ndcg']:.4f}"),
    ]
    tab2 = pd.DataFrame(rows_out, columns=[
        "Modelo", "Hiperparametros varridos (grade)",
        "Valor final escolhido", "NDCG@10 (validacao)"
    ])
    tab2.to_csv(os.path.join(PAPER_DIR, "tab2_hyperparams.csv"), index=False)
    print(f"  -> tab2_hyperparams.csv  ({len(tab2)} modelos)")
    for _, r in tab2.iterrows():
        print(f"     {r['Modelo']:25s}  NDCG@10 = {r['NDCG@10 (validacao)']}")

    return cb_best, knn_best_data, svd_best


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def try_refresh_live_data():
    """
    Busca dados frescos do banco e recomputa item_df, similarity_matrix e popularity.
    Retorna (item_df, sim_matrix, popularity) ou (None, None, None) se o DB falhar.
    """
    try:
        from training.training import load_items_from_db, compute_item_similarity, compute_popularity
        print("  Recomputando item_df a partir do banco...")
        item_df = load_items_from_db()
        if item_df is None or item_df.empty:
            return None, None, None
        print(f"  item_df recomputado: {len(item_df)} itens")

        print("  Recomputando similarity_matrix...")
        sim_matrix = compute_item_similarity(item_df)

        print("  Recomputando popularity...")
        popularity = compute_popularity()

        return item_df, sim_matrix, popularity
    except Exception as e:
        print(f"  [AVISO] Falha ao buscar dados frescos do banco: {e}")
        return None, None, None


def main():
    print("=" * 62)
    print("  AVALIAÇÃO DOS MODELOS DE RECOMENDAÇÃO DE JOIAS")
    print("=" * 62)
    print(f"  Saída: {OUTPUT_DIR}\n")

    # Busca dados frescos do banco; fallback para cache em disco
    print("[DADOS] Buscando dados atualizados do banco...")
    item_df, sim_matrix, popularity = try_refresh_live_data()

    if item_df is None:
        print("  DB indisponível — carregando artefatos salvos em disco...")
        item_df   = load_model(ITEM_FEATURES_PATH)
        sim_matrix = load_model(os.path.join(MODEL_CACHE_PATH, "similarity_matrix.pkl"))
        popularity = load_model(POPULARITY_PATH)
    else:
        print("  Dados frescos carregados com sucesso.")

    status = lambda v, extra="": f"OK ({extra})" if v is not None else "AUSENTE"
    print(f"\n[MODELOS] Status dos artefatos:")
    print(f"  item_df          : {status(item_df, f'{len(item_df)} itens' if item_df is not None else '')}")
    print(f"  similarity_matrix: {status(sim_matrix, f'{sim_matrix.shape}' if sim_matrix is not None else '')}")
    print(f"  popularity       : {status(popularity, f'{len(popularity)} itens' if popularity is not None else '')}")
    print(f"  xgb_model        : {status(load_model(os.path.join(MODEL_CACHE_PATH, 'xgb_model.pkl')))}")
    print(f"  kmeans_model     : {status(load_model(os.path.join(MODEL_CACHE_PATH, 'kmeans_model.pkl')))}")
    print(f"  knn_model        : {status(load_model(os.path.join(MODEL_CACHE_PATH, 'knn_model.pkl')))}")
    print(f"  svd_model        : {status(load_model(os.path.join(MODEL_CACHE_PATH, 'svd_model.pkl')))}")

    # Busca interações e itens brutos do banco
    print("\n[BANCO] Buscando interações e itens brutos...")
    interactions_df = try_get_interactions()
    raw_items_df = try_get_raw_items()
    if interactions_df is not None:
        print(f"  interactions: {len(interactions_df)} linhas  |  "
              f"usuários únicos: {interactions_df['user_id'].nunique()}")
    else:
        print("  Interações indisponíveis — gráficos dependentes serão omitidos/simplificados.")
    if raw_items_df is not None:
        print(f"  raw_items   : {len(raw_items_df)} itens")

    # Gera tabela de estatísticas do dataset (Tabela 1 do artigo)
    generate_dataset_table(raw_items_df, interactions_df)

    # Gera tabela de hiperparâmetros (Tabela 2 do artigo)
    cb_threshold, knn_best_data, svd_best_k = generate_hyperparams_table(item_df, interactions_df, sim_matrix)

    # Gera comparacao de modelos no teste (Tabela 3 + Fig. 1 do artigo)
    test_metrics = generate_model_comparison(item_df, interactions_df, sim_matrix,
                                             cb_threshold=cb_threshold, knn_data_override=knn_best_data, svd_n_factors=svd_best_k)
    plot_model_comparison(test_metrics)
    generate_hybrid_gain_table(test_metrics)
    plot_k_sensitivity(item_df, interactions_df, sim_matrix, cb_threshold=cb_threshold, knn_data_override=knn_best_data, svd_n_factors=svd_best_k)
    plot_kmeans_clusters(item_df, interactions_df, sim_matrix, cb_threshold=cb_threshold, knn_data_override=knn_best_data, svd_n_factors=svd_best_k)
    plot_cold_start(item_df, interactions_df, sim_matrix, cb_threshold=cb_threshold, knn_data_override=knn_best_data, svd_n_factors=svd_best_k)

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
