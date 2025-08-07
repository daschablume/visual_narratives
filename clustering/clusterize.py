import ast
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


def load_lst_from_saved_txt(df: pd.DataFrame, columns: list[str]):
    result = df[columns].copy()
    for col in columns:
        result[col] = result[col].apply(ast.literal_eval)
    return result


def replace_roles_with_clusterized_labels(
    column: str, resolved_df: pd.DataFrame, updated_roles_df: pd.DataFrame
):
    """
    Replace role values with clusterized labels based on sentence_id matching.
    
    Args:
        column (str): The name of the role column to be updated (e.g., 'ARG0').
        resolved_df (pd.DataFrame): DataFrame with clusterized labels and metadata.
            Columns: ['cluster_id', 'label', 'cluster_size', 'label_sentence_id',
                      'label_image_id', 'sentence_indices', 'image_indices']
        updated_roles_df (pd.DataFrame): Original roles DataFrame with sentence_id column.
    
    Returns:
        pd.DataFrame: Updated DataFrame with clusterized labels.
    """
    
    # Create mapping from sentence_id to cluster label
    sentence_id_to_label = {}
    
    for _, row in resolved_df.iterrows():
        label = row['label']
        sentence_indices = row['sentence_indices']
        
        for sentence_id in sentence_indices:
            sentence_id_to_label[sentence_id] = label
 
    mask = updated_roles_df['sentence_id'].isin(sentence_id_to_label.keys())
    updated_roles_df.loc[mask, column] = updated_roles_df.loc[
        mask, 'sentence_id'].map(sentence_id_to_label)
        
    return updated_roles_df


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


def create_manual_clustering_with_batches(
    phrases: pd.DataFrame, column_name, folder, batch_size=20000, threshold = 0.75
):
    """
    Create clusters of phrases based on their semantic similarity using batches.
    At first, creates a similarity matrix for each batch of phrases, then again 
        creates a similarity matrix for the clusterized batches.
    Args:
        phrases (pd.DataFrame): DataFrame containing phrases of the corresponding role 
            (like ARG0, etc) and their metadata: sentence_id, image_id.
        column_name (str): The column name in the DataFrame to cluster.
        folder (str): Folder to save the results.
        batch_size (int): Size of each batch for processing.
        threshold (float): Similarity threshold for clustering phrases.
    
    Returns:
        list: A list of clusters, where each cluster is a tuple containing:
            (cluster_id, label, cluster_size, label_sentence_id, label_image_id,
             sentence_indices, image_indices)
    
    TODO:
    1) star-shaped clusters (in principle)
    2) 
    """
    inner_folder = os.path.join(folder, column_name)
    os.makedirs(inner_folder, exist_ok=True)

    phrases_list = phrases[column_name].tolist()
    sentence_ids = phrases["sentence_id"].tolist()
    image_ids = phrases["image_id"].tolist()

    total_groups = []
    total_word_groups = []
    
    for batch_start in range(0, len(phrases), batch_size):
        print('**********************')
        print(
            f'Processing batch {batch_start // batch_size + 1} of '
            f'{(len(phrases) - 1) // batch_size + 1}'
        )
        
        batch_end = min(batch_start + batch_size, len(phrases))
        batch_phrases = phrases_list[batch_start:batch_end]
        vectors = EMBEDDING_MODEL.get_vectors(batch_phrases, progress_bar=True)
        
        print('Creating similarity matrix')
        sim_matrix = cosine_similarity(vectors)

        # Create mapping from vector index to phrase metadata
        vector_id2meta = {
            vec_idx: {
                "phrase_batch_id": vec_idx,
                "sentence_id": sentence_ids[batch_start + vec_idx],
                "image_id": image_ids[batch_start + vec_idx],
            }
            for vec_idx in range(len(batch_phrases))
        }

        groups = []
        visited = set()

        print('Grouping phrases')
        for vec_i in range(len(vectors)):
            if vec_i in visited:
                continue
            meta = vector_id2meta[vec_i]
            group = [meta]
            visited.add(vec_i)
            
            for vec_j in range(vec_i + 1, len(vectors)):  # Start from vec_i+1 to avoid duplicates
                if vec_j not in visited and sim_matrix[vec_i, vec_j] >= threshold:
                    group.append(vector_id2meta[vec_j])
                    visited.add(vec_j)
            
            groups.append(group)
        
        word_groups = []
        for group in groups:
            word_group = [batch_phrases[meta["phrase_batch_id"]] for meta in group]
            word_groups.append(word_group)

        total_word_groups.extend(word_groups)
        total_groups.extend(groups)
    
    print('Creating final similarity matrix')
    pre_cluster_id2phrases_ids = {idx: group for idx, group in enumerate(total_groups)}
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
        
        for pre_j in range(pre_i + 1, len(pre_clusters_vectors)):
            if pre_j not in visited and sim_matrix[pre_i, pre_j] >= threshold:
                clustered_group.append(pre_j)
                visited.add(pre_j)
        
        clusters_groups.append(clustered_group)

    # Create final clusters
    print('Creating final clusters')
    clusters = []
    for cl_id, cl_group in enumerate(clusters_groups):
        cl_phrases = []
        sentence_indices = []
        image_indices = []
        
        # Collect all phrases and metadata from pre-clusters
        for pre_cluster_id in cl_group:
            pre_cl_phrase_metas = pre_cluster_id2phrases_ids[pre_cluster_id]
            for meta in pre_cl_phrase_metas:
                sentence_indices.append(meta["sentence_id"])
                image_indices.append(meta["image_id"])
                cl_phrases.append(
                    phrases[
                        phrases['sentence_id'] == meta["sentence_id"]
                    ][column_name].values[0]
            )
        
        # Get most common phrase as cluster label
        cluster_label = Counter(cl_phrases).most_common(1)[0][0]
        label_phrase_idx = cl_phrases.index(cluster_label)
        label_sentence_id = sentence_indices[label_phrase_idx]
        label_image_id = image_indices[label_phrase_idx]

        clusters.append((
            cl_id, cluster_label, len(cl_phrases),
            label_sentence_id, label_image_id,
            sentence_indices, image_indices
        ))

    # Sort by size
    print('Sorting clusters')
    clusters.sort(key=lambda x: x[2], reverse=True)
    
    # Save cluster labels to CSV
    print('Saving clusters')
    cluster2label_path = os.path.join(inner_folder, 'cluster_label_mapping.csv')
    df = pd.DataFrame([
        {
            "cluster_id": idx,
            "label": label,
            "cluster_size": size,
            "label_sentence_id": sentence_id,
            "label_image_id": image_id,
            "sentence_indices": sentence_indices,
            "image_indices": image_indices
        }
        for idx, label, size, sentence_id, image_id, sentence_indices, image_indices in clusters
    ])
    df.to_csv(cluster2label_path, index=False, sep=";")

    # Save all the words from a cluster to text file
    print('Saving clusters to text file')
    clusters2words_path = os.path.join(inner_folder, 'clusters.txt')
    with open(clusters2words_path, 'w') as f:
        for idx, label, size, _, _, sentence_indices, image_indices in clusters:
            f.write(f"CLUSTER{idx}\t{label}\t(size: {size})\n")
            # save all the roles of the cluster
            clusterized_labels = phrases[
                phrases['sentence_id'].isin(sentence_indices)][column_name].tolist()
            f.write(f"Labels: {', '.join(clusterized_labels)}\n")

    # Save cluster2indices mapping to JSON
    print('Saving cluster2index mapping')
    cluster_index2indices_path = os.path.join(inner_folder, 'cluster_index_to_indices.json')
    cluster_dict = {
        idx: {
            "sentence_indices": sentence_indices, 
            "image_indices": image_indices
        }
        for idx, _, _, _, _, sentence_indices, image_indices in clusters
    }
    with open(cluster_index2indices_path, 'w') as f:
        json.dump(cluster_dict, f, indent=2)
    
    print(f'Clustering completed. Created {len(clusters)} clusters.')
    return clusters
