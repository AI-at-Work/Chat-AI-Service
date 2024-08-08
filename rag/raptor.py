import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import umap.umap_ as umap
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import prompts.rag_prompt


def global_cluster_embeddings(embeddings: np.ndarray, dim: int, n_neighbors: Optional[int] = None,
                              metric: str = "cosine") -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=dim,
        metric=metric,
        random_state=None,  # Remove random_state
        n_jobs=-1  # Use all available cores
    ).fit_transform(embeddings)


def local_cluster_embeddings(embeddings: np.ndarray, dim: int, num_neighbors: int = 10,
                             metric: str = "cosine") -> np.ndarray:
    return umap.UMAP(
        n_neighbors=num_neighbors,
        n_components=dim,
        metric=metric,
        random_state=None,  # Remove random_state
        n_jobs=-1  # Use all available cores
    ).fit_transform(embeddings)


def get_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 50) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)

    def compute_bic(n):
        gm = GaussianMixture(n_components=n, random_state=None)  # Remove random_state
        gm.fit(embeddings)
        return gm.bic(embeddings)

    with ThreadPoolExecutor() as executor:
        bics = list(executor.map(compute_bic, n_clusters))

    return n_clusters[np.argmin(bics)]


def GMM_cluster(embeddings: np.ndarray, threshold: float):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=None)  # Remove random_state
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def process_global_cluster(args):
    i, embeddings, global_clusters, dim, threshold, total_clusters = args
    global_cluster_embeddings_ = embeddings[np.array([i in gc for gc in global_clusters])]

    if len(global_cluster_embeddings_) == 0:
        return [], 0
    if len(global_cluster_embeddings_) <= dim + 1:
        return [(idx, np.array([total_clusters])) for idx in range(len(global_cluster_embeddings_))], 1

    reduced_embeddings_local = local_cluster_embeddings(global_cluster_embeddings_, dim)
    local_clusters, n_local_clusters = GMM_cluster(reduced_embeddings_local, threshold)

    results = []
    for j in range(n_local_clusters):
        local_cluster_embeddings_ = global_cluster_embeddings_[np.array([j in lc for lc in local_clusters])]
        indices = np.where((embeddings == local_cluster_embeddings_[:, None]).all(-1))[1]
        results.extend([(idx, np.array([j + total_clusters])) for idx in indices])

    return results, n_local_clusters


def perform_clustering(embeddings: np.ndarray, dim: int, threshold: float) -> List[np.ndarray]:
    if len(embeddings) <= dim + 1:
        return [np.array([0]) for _ in range(len(embeddings))]

    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    global_clusters, n_global_clusters = GMM_cluster(reduced_embeddings_global, threshold)

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    with ProcessPoolExecutor() as executor:
        args_list = [(i, embeddings, global_clusters, dim, threshold, total_clusters) for i in range(n_global_clusters)]
        all_results = list(executor.map(process_global_cluster, args_list))

    for results, n_local_clusters in all_results:
        for idx, cluster in results:
            all_local_clusters[idx] = np.append(all_local_clusters[idx], cluster + total_clusters)
        total_clusters += n_local_clusters

    return all_local_clusters


def embed(texts, embd):
    with ThreadPoolExecutor() as executor:
        text_embeddings = list(executor.map(embd.encode, texts))
    return np.array(text_embeddings)


def embed_cluster_texts(texts, embd):
    text_embeddings_np = embed(texts, embd)
    cluster_labels = perform_clustering(text_embeddings_np, 10, 0.1)
    return pd.DataFrame({
        "text": texts,
        "embd": list(text_embeddings_np),
        "cluster": cluster_labels
    })


def fmt_txt(df: pd.DataFrame) -> str:
    return "--- --- \n --- --- ".join(df["text"].tolist())


def process_cluster(cluster_data, llm_service, provider, model):
    i, df_cluster = cluster_data
    formatted_txt = fmt_txt(df_cluster)
    prompt = prompts.rag_prompt.get_question_context_prompt(
        question="Give a detailed summary of the documentation provided.",
        context=f"Here is a sub-set of doc. \n\nDocumentation:\n{formatted_txt}")
    summary = llm_service.generate(provider, model, prompt, "")
    return i, summary


def embed_cluster_summarize_texts(texts: List[str], llm_service, provider, model, embd, level: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_clusters = embed_cluster_texts(texts, embd)

    expanded_list = []
    for index, row in df_clusters.iterrows():
        expanded_list.extend(
            [{"text": row["text"], "embd": row["embd"], "cluster": cluster} for cluster in row["cluster"]])

    expanded_df = pd.DataFrame(expanded_list)
    all_clusters = expanded_df["cluster"].unique()

    print(f"--Generated {len(all_clusters)} clusters--")

    with ProcessPoolExecutor() as executor:
        cluster_summaries = list(executor.map(
            process_cluster,
            [(i, expanded_df[expanded_df["cluster"] == i]) for i in all_clusters],
            [llm_service, provider, model] * len(all_clusters)
        ))

    summaries = [summary for _, summary in sorted(cluster_summaries)]

    df_summary = pd.DataFrame({
        "summaries": summaries,
        "level": [level] * len(summaries),
        "cluster": list(all_clusters),
    })

    return df_clusters, df_summary


def recursive_embed_cluster_summarize(texts: List[str], llm_service, provider, model, embd, level: int = 1, n_levels: int = 3) -> Dict[
    int, Tuple[pd.DataFrame, pd.DataFrame]]:
    results = {}
    df_clusters, df_summary = embed_cluster_summarize_texts(texts, llm_service, provider, model, embd, level)
    results[level] = (df_clusters, df_summary)

    unique_clusters = df_summary["cluster"].nunique()
    if level < n_levels and unique_clusters > 1:
        new_texts = df_summary["summaries"].tolist()
        next_level_results = recursive_embed_cluster_summarize(new_texts, llm_service, provider, model, embd, level + 1, n_levels)
        results.update(next_level_results)

    return results
