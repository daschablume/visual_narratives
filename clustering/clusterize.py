import ast
from collections import Counter
import json
import os

import numpy as np
import pandas as pd
from relatio.embeddings import Embeddings
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# TODO: improve lemmatization; now I call spacy for each word, which is slow
NLP = spacy.load("en_core_web_sm")

# TODO: everywhere rename "id" back to "image_id", more clear
# TODO: get rid of Relatio
EMBEDDING_MODEL = Embeddings(
    embeddings_type="SentenceTransformer",
    embeddings_model='all-MiniLM-L6-v2',
)


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


def replace_synt_parsed_with_clusterized_labels(
    clusters_df: pd.DataFrame, sentences_df: pd.DataFrame
):
    # TODO: .iterrows() is slow, consider making it better
    # i.e. finish replace_synt_parsed_with_clusterized_labels_altern
    for _, row in clusters_df.iterrows():
        subset = sentences_df[sentences_df["sentence_id"].isin(row['sentence_indices'])]
        for position_id, sentence_id in zip(row['position_indices'], row['sentence_indices']):
            orig_sentence = subset[subset['sentence_id'] == sentence_id]['parsed_sentence'].values[0]
            orig_sentence[position_id] = row['label']
            sentences_df.at[
                sentences_df.index[sentences_df['sentence_id'] == sentence_id][0], 
                'parsed_sentence'
            ] = orig_sentence
    return sentences_df


def replace_synt_parsed_with_clusterized_labels_altern(clusters_df, sentences_df):
    raise NotImplementedError('This function is not finished')
    # flatten clusters_df: each element => one row
    assignments = (
        clusters_df
        .explode(['sentence_indices', 'position_indices'])
        .rename(columns={
            'sentence_indices': 'sentence_id',
            'position_indices': 'position_id'
        })
        [['sentence_id', 'position_id', 'label']]
    )

    # merge with sentences_df
    merged = assignments.merge(
        sentences_df[['id', 'parsed_labels', 'parsed_sentence']],
        on='sentence_id',
        how='left')
    merged['parsed_sentence'] = merged.apply(
        lambda row: [
            row['label'] if i == row['position_id'] else word
            for i, word in enumerate(row['parsed_sentence'])
        ],
        axis=1
    )
    # TODO: return all the rows, not just the updated ones


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
    phrases: pd.DataFrame, column_name, folder, batch_size=20000, threshold = 0.7
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
    2) do NOT drag all the metadata thru all the steps of clustering -- unnecessary
    3) centroid instead of the most common phrase as cluster label?
    4) do not save label_image_id and phrase_id, cause that's just the first occuring, no sense
    5) cluster2index file is unnecessary
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


def prepare_synt_df_to_clustering(df):
    #df[['parsed_labels', 'parsed_sentence']] = load_lst_from_saved_txt(
            #df, ['parsed_labels', 'parsed_sentence'])
    verbs_meta = []
    np_meta = []
    for _, row in df.iterrows():
        labels, sentence = row['parsed_labels'], row['parsed_sentence']

        for idx in range(len(labels)):
            label, part = labels[idx], sentence[idx]
            if label.startswith('V'):  # verb
                lemmatized = NLP(part)[0].lemma_
                verbs_meta.append({
                    'word': lemmatized,
                    'sentence_id': row['sentence_id'],
                    'image_id': row['id'],
                    'position_idx': idx
                })
            elif label.startswith('NN') or label == 'NP':  # noun or noun phrase
                np_meta.append({
                    'word': part,
                    'sentence_id': row['sentence_id'],
                    'image_id': row['id'],
                    'position_idx': idx
                })
    return verbs_meta, np_meta
            

def clusterize_synt_parsed(
    phrases_meta: list[dict], save_folder_path: str,
    threshold=0.7, batch_size=20000
):
    '''
    Args:
        phrases_meta: a list of dicts like this:
            {"word": "People", "sentence_id": "NP", "image_id", "position_idx": 0}
    
    TODO: 
    1) star-shaped clusters => explore graph methods
    2) centroid instead of most common phrase as cluster label?
    '''
    phrases = [phm['word'] for phm in phrases_meta]
    vectors = EMBEDDING_MODEL.get_vectors(phrases, progress_bar=True)

    assert len(vectors) == len(phrases)

    all_batches_groups = []
    for batch_start in range(0, len(phrases), batch_size):
        print('**********************')
        print(
            f'Processing batch {batch_start // batch_size + 1} of '
            f'{(len(phrases) - 1) // batch_size + 1}'
        )
        
        batch_end = min(batch_start + batch_size, len(phrases))
        batch_vectors = vectors[batch_start:batch_end]
        batch_vector_id2vector_id = {i:i+batch_start for i in range(len(batch_vectors))}

        sim_matrix = cosine_similarity(batch_vectors)

        batch_groups = []
        visited = set()

        # group vectors based on similarity matrix
        print('Grouping vectors')
        for vec_i in range(len(batch_vectors)):
            if vec_i in visited:
                continue
            visited.add(vec_i)
            batch_group = [vec_i]
            
            for vec_j in range(vec_i + 1, len(batch_vectors)):  # vec_1 + 1 to not compare what has been already compared
                if vec_j not in visited and sim_matrix[vec_i, vec_j] >= threshold:
                    batch_group.append(vec_j)
                    visited.add(vec_j)
            
            batch_groups.append(batch_group)
        
        # remap group indices for a specific batch to original indices
        remapped_groups = [
            [batch_vector_id2vector_id[idx] for idx in group] 
            for group in batch_groups
        ]
        all_batches_groups.extend(remapped_groups)
    
    print('Creating final similarity matrix')
    pre_cluster_id2phrases_ids = {idx: group for idx, group in enumerate(all_batches_groups)}
    # pick a label for each pre-cluster
    pre_cluster_id2cluster_name = {
        idx: Counter([phrases[phrase_idx] for phrase_idx in group]).most_common(1)[0][0] 
        for idx, group in enumerate(all_batches_groups)
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
        position_indices = []
        
        # Collect all phrases and metadata from pre-clusters
        for pre_cluster_id in cl_group:
            phrase_ids = pre_cluster_id2phrases_ids[pre_cluster_id]
            cl_phrases.extend([phrases_meta[i]["word"] for i in phrase_ids])
            sentence_indices.extend([phrases_meta[i]['sentence_id'] for i in phrase_ids])
            image_indices.extend([phrases_meta[i]['image_id'] for i in phrase_ids])
            position_indices.extend([phrases_meta[i]['position_idx'] for i in phrase_ids])
        
        # Get most common phrase as cluster label
        cl_label = Counter(cl_phrases).most_common(1)[0][0]

        clusters.append((
            cl_id, cl_label, len(cl_phrases), cl_phrases,
            sentence_indices, image_indices, position_indices
        ))

    # Sort by size
    print('Sorting clusters')
    clusters.sort(key=lambda x: x[2], reverse=True)
    
    # Save clusters to CSV
    print('Saving clusters')
    cluster2label_path = os.path.join(save_folder_path, 'clusters.csv')
    df = pd.DataFrame([
        {
            "cluster_id": idx,
            "label": label,
            "cluster_size": size,
            "sentence_indices": sentence_indices,
            "image_indices": image_indices,
            "position_indices": position_indices
        }
        for idx, label, size, _, sentence_indices, image_indices, position_indices in clusters
    ])
    df.to_csv(cluster2label_path, index=False, sep=";")

    # Save all the words from a cluster to text file
    print('Saving clusters to text file')
    clusters2words_path = os.path.join(save_folder_path, 'clusters.txt')
    with open(clusters2words_path, 'w') as f:
        for idx, label, size, cl_labels, _, _, _ in clusters:
            f.write(f"CLUSTER{idx}\t{label}\t(size: {size})\n")
            # save all the roles of the cluster
            f.write(f"Labels: {', '.join(cl_labels)}\n")
    
    print(f'Clustering completed. Created {len(clusters)} clusters.')
