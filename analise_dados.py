import os
import sys
import time
import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, Birch, MeanShift
from sklearn.metrics import silhouette_score, f1_score, adjusted_rand_score, normalized_mutual_info_score
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import colorama

colorama.init(autoreset=True)


# ============================================================
# UTILITÁRIOS
# ============================================================

def log(msg, tipo="info"):
    cores = {"info": colorama.Fore.CYAN,
             "ok": colorama.Fore.GREEN,
             "warn": colorama.Fore.YELLOW,
             "erro": colorama.Fore.RED}
    print(cores.get(tipo, "") + msg + colorama.Style.RESET_ALL)


def tempo_exec(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        resultado = func(*args, **kwargs)
        log(f"⏱️ Tempo: {time.time() - t0:.2f}s", "warn")
        return resultado

    return wrapper


# ============================================================
# CARREGAMENTO DE SEQUÊNCIAS
# ============================================================

@tempo_exec
def carregar_sequencias(arquivos):
    sequencias, rotulos = [], []
    for arquivo in arquivos:
        if not os.path.exists(arquivo):
            sys.exit(f"ERRO: Arquivo não encontrado: {arquivo}")
        log(f"Lendo {arquivo} ...")
        for record in SeqIO.parse(arquivo, "fasta"):
            sequencias.append(str(record.seq))
            rotulos.append(record.description.split()[0])
    log(f"Total de sequências: {len(sequencias)}", "ok")
    return sequencias, np.array(rotulos)


# ============================================================
# EXTRAÇÃO DE KMERS
# ============================================================

def gerar_kmers_seq(seq, k=2, skip=1):
    return [seq[i] + seq[i + 1 + skip] for i in range(len(seq) - (k - 1) - skip)]


@tempo_exec
def extrair_kmers_unicos(sequencias, k=2, skip=1, n_jobs=-1):
    log("Extraindo todos os kmers únicos...", "info")
    kmers_list = Parallel(n_jobs=n_jobs)(delayed(gerar_kmers_seq)(seq, k, skip) for seq in sequencias)
    kmers_unicos = sorted(set(k for lista in kmers_list for k in lista))
    log(f"Total de kmers únicos: {len(kmers_unicos)}", "ok")
    return kmers_unicos, kmers_list


@tempo_exec
def construir_matriz_sparse(kmers_unicos, kmers_list):
    log("Construindo matriz esparsa binária...", "info")
    n_seq = len(kmers_list)
    n_kmers = len(kmers_unicos)
    matriz = lil_matrix((n_seq, n_kmers), dtype=np.uint8)
    kmer_idx = {k: i for i, k in enumerate(kmers_unicos)}

    for i, kmers in enumerate(kmers_list):
        for k in set(kmers):
            matriz[i, kmer_idx[k]] = 1

    log(f"Matriz {n_seq}x{n_kmers} construída (esparsa)", "ok")
    return csr_matrix(matriz)


# ============================================================
# PCA
# ============================================================

@tempo_exec
def aplicar_pca(matriz, n_componentes=300):
    log("Aplicando PCA...", "info")
    scaler = StandardScaler(with_mean=False)  # compatível com sparse
    dados_norm = scaler.fit_transform(matriz)
    pca = PCA(n_components=min(n_componentes, matriz.shape[1]), random_state=42)
    componentes = pca.fit_transform(dados_norm)
    df_pca = pd.DataFrame(componentes, columns=[f"PC{i + 1}" for i in range(componentes.shape[1])])
    log("PCA concluída.", "ok")
    return df_pca, pca.explained_variance_ratio_


# ============================================================
# CLUSTERING
# ============================================================

@tempo_exec
def testar_todos_algoritmos(df_pca, rotulos_reais, sample_size=1000):
    log("Rodando algoritmos de clustering...", "info")
    le = LabelEncoder()
    rotulos_num = le.fit_transform(rotulos_reais)
    usar_metricas_externas = len(np.unique(rotulos_num)) <= 0.5 * len(rotulos_num)

    algoritmos = {
        "KMeans": KMeans(n_clusters=5, random_state=42, n_init='auto'),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5, n_jobs=-1),
        "Agglomerative": AgglomerativeClustering(n_clusters=5, linkage='ward', compute_full_tree=False),
        "Spectral": SpectralClustering(n_clusters=5, random_state=42, assign_labels="discretize", n_jobs=-1,
                                       affinity="nearest_neighbors"),
        "Birch": Birch(n_clusters=5, threshold=0.5),
        "MeanShift": MeanShift(bin_seeding=True, cluster_all=True, min_bin_freq=2)
    }

    resultados = []

    for nome, modelo in algoritmos.items():
        try:
            log(f"Executando {nome}...", "info")
            df_input = df_pca
            if nome in ["Spectral", "Agglomerative", "Birch"] and len(df_pca) > 2000:
                idx = np.random.choice(len(df_pca), 2000, replace=False)
                df_input = df_pca.iloc[idx]
                log(f"Subamostragem para {nome}: {len(df_input)} amostras", "warn")

            pred = modelo.fit_predict(df_input)

            if df_input.shape[0] < df_pca.shape[0]:
                full_pred = np.full(df_pca.shape[0], -1, dtype=int)
                full_pred[idx] = pred
                pred = full_pred

            if len(set(pred)) < 2:
                log(f"{nome}: menos de 2 clusters, ignorando métricas.", "warn")
                continue

            n = len(pred)
            if n > sample_size:
                idx_sil = np.random.choice(n, sample_size, replace=False)
                sil = silhouette_score(df_pca.iloc[idx_sil], pred[idx_sil])
            else:
                sil = silhouette_score(df_pca, pred)

            if usar_metricas_externas:
                valid_idx = pred != -1
                f1 = f1_score(rotulos_num[valid_idx], pred[valid_idx],
                              average="weighted") if valid_idx.any() else np.nan
                ari = adjusted_rand_score(rotulos_num[valid_idx], pred[valid_idx]) if valid_idx.any() else np.nan
                nmi = normalized_mutual_info_score(rotulos_num[valid_idx],
                                                   pred[valid_idx]) if valid_idx.any() else np.nan
            else:
                f1 = ari = nmi = np.nan

            resultados.append({
                "Algoritmo": nome,
                "Silhouette": sil,
                "F1": f1,
                "ARI": ari,
                "NMI": nmi
            })

        except Exception as e:
            log(f"Erro em {nome}: {e}", "erro")
            continue

    df_resultados = pd.DataFrame(resultados)
    log("Resultados de clustering concluídos.", "ok")
    print(df_resultados)

    if usar_metricas_externas and not df_resultados.empty:
        corr = df_resultados.corr(numeric_only=True)
        print("\n📈 Correlação entre métricas:")
        print(corr["F1"].sort_values(ascending=False))
    else:
        corr = None

    return df_resultados, corr


# ============================================================
# SELEÇÃO DE MELHOR K (KMEANS)
# ============================================================

@tempo_exec
def escolher_melhor_k(df_pca, rotulos_reais, max_k=15):
    log("Avaliando variação de k para KMeans...", "info")
    le = LabelEncoder()
    rotulos_num = le.fit_transform(rotulos_reais)
    usar_metricas_externas = len(np.unique(rotulos_num)) <= 0.5 * len(rotulos_num)

    resultados = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        pred = km.fit_predict(df_pca)
        sil = silhouette_score(df_pca, pred)
        f1 = f1_score(rotulos_num, pred, average='weighted') if usar_metricas_externas else np.nan
        resultados.append((k, sil, f1))

    df = pd.DataFrame(resultados, columns=["k", "Silhouette", "F1"])
    melhor = df.loc[df["Silhouette"].idxmax()]
    log(f"Melhor k={melhor.k} (Silhouette={melhor.Silhouette:.3f})", "ok")

    plt.figure(figsize=(7, 5))
    plt.plot(df["k"], df["Silhouette"], 'bo-', label="Silhouette")
    if usar_metricas_externas:
        plt.plot(df["k"], df["F1"], 'ro--', label="F1")
    plt.title("Variação de k (KMeans)")
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Métrica")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return int(melhor.k)


# ============================================================
# CORRELAÇÃO ENTRE PCs E CLUSTERS
# ============================================================

@tempo_exec
def correlacionar_300_pcs(df_pca, rotulos_pred, salvar_csv=True):
    log("Calculando correlação entre 300 PCs e clusters...", "info")
    df_corr = df_pca.copy()
    df_corr["Cluster"] = rotulos_pred

    corr_values = df_corr.corr(numeric_only=True)["Cluster"].drop("Cluster", errors="ignore")
    corr_table = pd.DataFrame({
        "Componente": corr_values.index,
        "Correlação_Cluster": corr_values.values
    })
    corr_table["Abs_Correlação"] = corr_table["Correlação_Cluster"].abs()
    corr_table = corr_table.sort_values(by="Abs_Correlação", ascending=False).reset_index(drop=True)

    if salvar_csv:
        corr_table.to_csv("correlacao_300_pcs_clusters.csv", index=False)
        log("Tabela salva em 'correlacao_300_pcs_clusters.csv'.", "ok")

    plt.figure(figsize=(6, 8))
    sns.heatmap(
        corr_table.head(20).set_index("Componente")[["Correlação_Cluster"]],
        annot=True, cmap="coolwarm", cbar=True, fmt=".3f"
    )
    plt.title("Top 20 PCs mais correlacionados com Clusters")
    plt.tight_layout()
    plt.show()

    print("\n📊 Top 10 PCs mais correlacionados:")
    print(corr_table.head(10))
    return corr_table


# ============================================================
# VISUALIZAÇÃO DE CLUSTERS
# ============================================================

def plot_clusters(df_pca, rotulos_pred, k, silhouette):
    plt.figure(figsize=(8, 6))
    cores = plt.cm.tab20(np.linspace(0, 1, k))
    for cid in range(k):
        plt.scatter(
            df_pca.loc[rotulos_pred == cid, "PC1"],
            df_pca.loc[rotulos_pred == cid, "PC2"],
            s=40, color=cores[cid], alpha=0.7, label=f"Cluster {cid}"
        )
    plt.title(f"Clusters KMeans (k={k}) - PCA\nSilhouette={silhouette:.2f}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

if __name__ == "__main__":
    arquivos = ["sequencia_proteinas_40.txt", "sequencia_proteinas_95.txt"]

    sequencias, rotulos_reais = carregar_sequencias(arquivos)
    kmers_unicos, kmers_list = extrair_kmers_unicos(sequencias, k=2, skip=1)
    matriz_sparse = construir_matriz_sparse(kmers_unicos, kmers_list)
    df_pca, variancias = aplicar_pca(matriz_sparse, n_componentes=300)

    df_resultados, corr = testar_todos_algoritmos(df_pca, rotulos_reais)
    melhor_k = escolher_melhor_k(df_pca, rotulos_reais)

    km = KMeans(n_clusters=melhor_k, random_state=42, n_init='auto')
    rotulos_pred = km.fit_predict(df_pca)
    sil = silhouette_score(df_pca, rotulos_pred)

    plot_clusters(df_pca, rotulos_pred, melhor_k, sil)
    correlacionar_300_pcs(df_pca, rotulos_pred)
