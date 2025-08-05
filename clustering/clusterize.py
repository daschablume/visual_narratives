from collections import Counter
import json
import os

import numpy as np
import pandas as pd
from relatio.embeddings import Embeddings
from sklearn.metrics.pairwise import cosine_similarity


EMBEDDING_MODEL = Embeddings(
    embeddings_type="SentenceTransformer",
    embeddings_model='all-MiniLM-L6-v2',
)


def create_manual_clustering(phrases, role, folder):
    inner_folder = os.path.join(folder, role)
    os.makedirs(inner_folder, exist_ok=True)
    vectors = EMBEDDING_MODEL.get_vectors(phrases, progress_bar=True)

    # Compute full similarity matrix
    sim_matrix = cosine_similarity(vectors)

    threshold = 0.8
    groups = []
    visited = set()

    for i in range(len(vectors)):
        if i in visited:
            continue
        group = [i]
        visited.add(i)
        for j in range(len(vectors)):
            if j not in visited and sim_matrix[i, j] >= threshold:
                group.append(j)
                visited.add(j)
        
        groups.append(group)

    word_groups = [[phrases[i] for i in group] for group in groups]
    #sorted_groups = sorted(groups, key=len, reverse=True)[:30]
    #sorted_word_groups = [[narratives_clean[i] for i in group] for group in sorted_groups]

    clusters = []
    # preserve indexes because we need to sort them
    for idx in range(len(word_groups)):
        word_group = word_groups[idx]
        group = groups[idx]
        most_common = Counter(word_group).most_common(1)[0][0]
        clusters.append((idx, most_common, len(word_group), word_group, group))

    # sort by size
    clusters.sort(key=lambda x: x[2], reverse=True)

    # save cluster labels to CSV
    cluster2label_path = os.path.join(inner_folder, 'cluster_label_mapping.csv')
    df = pd.DataFrame([
        {"index": idx, "word": label, "cluster size": size, "cluster_indeces": [idx]}
        for idx, label, size, _, _ in clusters
    ])
    if os.path.exists(cluster2label_path):
        df.to_csv(cluster2label_path, index=False, sep=";", mode='a', header=False)
    else:
        df.to_csv(cluster2label_path, index=False, sep=";")

    # save all the words from a cluster to text file
    clusters2words_path = os.path.join(inner_folder, 'clusters.txt')
    mode = 'a' if os.path.exists(clusters2words_path) else 'w'
    with open(clusters2words_path, mode) as f:
        for idx, label, _, word_group, _ in clusters:
            f.write(f"CLUSTER{idx}\t{label}\n")
            f.write(", ".join(word_group) + "\n\n")

    # save cluster2indexes mapping to JSON
    cluster_index2indices_path = os.path.join(inner_folder, 'cluster_index_to_indices.json')
    cluster_dict = {idx: indices for idx, _, _, _, indices in clusters}
    if os.path.exists(cluster_index2indices_path):
        with open(cluster_index2indices_path) as f:
            existing = json.load(f)
            existing.update(cluster_dict)
    else: 
        existing = cluster_dict
    with open(cluster_index2indices_path, 'w') as f:
        json.dump(existing, f, indent=2)
            
    return clusters


def replace_roles_with_clusterized_labels(column, resolved_df, roles_df, folder):
    
    # set all values in this column to nan to remove deleted entities
    roles_df[column] = np.nan
    
    cluster_index2indices_path = os.path.join(folder, column, 'cluster_index_to_indices.json')
    with open(cluster_index2indices_path, "r") as f:
        cluster_index2indices = json.load(f)

    # only process clusters that remain in resolved_df
    for _, row in resolved_df.iterrows():
        cluster_idx = row['index']
        label = row['word']
        merge_with = row['cluster_indices']

        if cluster_idx not in merge_with:
            merge_with.append(cluster_idx)
        
        all_indices = []
        for idx in merge_with:
            indices = cluster_index2indices[str(idx)]
            all_indices.extend(indices)

        for i in all_indices:
            roles_df.loc[i, column] = label
    
    return roles_df


def prettify(df_row) -> str:
    """
    Takes a narrative statement dictionary and returns a pretty string.

    Args:
        narrative: a dictionary with the following keys: "ARG0", "B-V", "B-ARGM-NEG", "B-ARGM-MOD", "ARG1", "ARG2"

    Returns:
        a concatenated string of text
    """

    ARG0 = df_row['ARG0']
    V = df_row["B-V"]
    NEG = df_row["B-ARGM-NEG"]
    if NEG is not np.nan:
        NEG = "not"

    MOD = df_row["B-ARGM-MOD"]
    ARG1 = df_row["ARG1"]
    ARG2 = df_row["ARG2"]

    pretty_narrative = (ARG0, MOD, NEG, V, ARG1, ARG2)


    pretty_narrative = " ".join([t for t in pretty_narrative if (t != "" and t is not np.nan)])

    return pretty_narrative


def get_roles_df(folder):
    roles_path = os.path.join(folder, 'postproc_roles.json')
    with open(roles_path) as f:
        srl_roles = json.load(f)

    return pd.DataFrame(srl_roles)


def create_manual_clustering_with_batches(phrases, role, folder, batch_size=20000):
    inner_folder = os.path.join(folder, role)
    os.makedirs(inner_folder, exist_ok=True)
    total_groups = []
    total_word_groups = []
    
    for batch_start in range(0, len(phrases), batch_size):
        print('**********************')
        print(
            f'Processing batch {batch_start // batch_size + 1} of '
            '{(len(phrases) - 1) // batch_size + 1}'
        )
        
        batch_end = min(batch_start + batch_size, len(phrases))
        batch = phrases[batch_start:batch_end]
        vectors = EMBEDDING_MODEL.get_vectors(batch, progress_bar=True)
        
        print('Creating similarity matrix')
        sim_matrix = cosine_similarity(vectors)

        # Create mapping from vector index to phrase index
        vector_id2phrase_id = {vec_idx: batch_start + vec_idx for vec_idx in range(len(batch))}

        threshold = 0.8
        groups = []
        visited = set()

        # Group phrases that cross the similarity threshold
        print('Grouping phrases')
        for vec_i in range(len(vectors)):  # Renamed to avoid collision
            if vec_i in visited:
                continue
            group_phrase_id = vector_id2phrase_id[vec_i]
            group = [group_phrase_id]
            visited.add(vec_i)
            
            for vec_j in range(len(vectors)):  
                if vec_j not in visited and sim_matrix[vec_i, vec_j] >= threshold:
                    group_phrase_id = vector_id2phrase_id[vec_j]
                    group.append(group_phrase_id)
                    visited.add(vec_j)
            
            groups.append(group)
        
        word_groups = [[phrases[phrase_id] for phrase_id in group] for group in groups]

        total_word_groups.extend(word_groups)
        total_groups.extend(groups)
    
    print('Creating final similarity matrix')
    pre_cluster_id2phrases_ids = {idx: phrase_ids for idx, phrase_ids in enumerate(total_groups)}
    pre_cluster_id2cluster_name = {
        idx: Counter(word_group).most_common(1)[0][0] 
        for idx, word_group in enumerate(total_word_groups)
    }
    
    pre_clusters_vectors = EMBEDDING_MODEL.get_vectors(
        list(pre_cluster_id2cluster_name.values()), progress_bar=True
    )
    sim_matrix = cosine_similarity(pre_clusters_vectors)

    clusters_groups = []
    visited = set()

    for pre_i in range(len(pre_clusters_vectors)):  
        if pre_i in visited:
            continue
        clustered_group = [pre_i]
        visited.add(pre_i)
        
        for pre_j in range(len(pre_clusters_vectors)):  
            if pre_j not in visited and sim_matrix[pre_i, pre_j] >= threshold:
                clustered_group.append(pre_j)
                visited.add(pre_j)
        
        clusters_groups.append(clustered_group)

    # Create final clusters
    print('Creating final clusters')
    clusters = []
    for cl_id, cl_group in enumerate(clusters_groups):
        cl_phrases_ids = []
        cl_phrases = []
        for pre_cluster_id in cl_group:
            # Fix: pre_cluster_id2phrases_ids contains lists of phrase IDs
            pre_cl_phrases_ids = pre_cluster_id2phrases_ids[pre_cluster_id]
            pre_cl_phrases = [phrases[ph_id] for ph_id in pre_cl_phrases_ids]
            cl_phrases_ids.extend(pre_cl_phrases_ids)
            cl_phrases.extend(pre_cl_phrases)
            
        cluster_label = Counter(cl_phrases).most_common(1)[0][0]
        clusters.append((cl_id, cluster_label, len(cl_phrases), cl_phrases, cl_phrases_ids))

    # Sort by size
    print('Sorting clusters')
    clusters.sort(key=lambda x: x[2], reverse=True)
    
    # save cluster labels to CSV
    print('Saving clusters')
    cluster2label_path = os.path.join(inner_folder, 'cluster_label_mapping.csv')
    df = pd.DataFrame([
        {"index": idx, "word": label, "cluster size": size, "cluster_indeces": [idx]}
        for idx, label, size, _, _ in clusters
    ])
    df.to_csv(cluster2label_path, index=False, sep=";")

    # save all the words from a cluster to text file
    print('Saving clusters to text file')
    clusters2words_path = os.path.join(inner_folder, 'clusters.txt')
    with open(clusters2words_path, 'w') as f:
        for idx, label, _, word_group, _ in clusters:
            f.write(f"CLUSTER{idx}\t{label}\n")
            f.write(", ".join(word_group) + "\n\n")

    # save cluster2indexes mapping to JSON
    print('Saving cluster2index mapping')
    cluster_index2indices_path = os.path.join(inner_folder, 'cluster_index_to_indices.json')
    cluster_dict = {idx: indices for idx, _, _, _, indices in clusters}
    with open(cluster_index2indices_path, 'w') as f:
        json.dump(cluster_dict, f, indent=2)
    

