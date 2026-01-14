import ast
from collections import defaultdict
import csv
import os

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from .embeddings import Embeddings


EMBEDDING_MODEL = Embeddings(normalize=True)    


def prepare_df_to_clustering(df: pd.DataFrame, verbs_csv_path=None, np_csv_path=None):
    '''
    Creates CSV files for verbs and noun phrases from the DataFrame for clustering.
    Saves data item by item to avoid memory issues with large datasets.
    '''
    if verbs_csv_path is None:
        verbs_csv_path = 'verbs_meta.csv'
        print('No verbs_csv_path provided, using default: verbs_meta.csv')
    if np_csv_path is None:
        np_csv_path = 'np_meta.csv'
        print('No np_csv_path provided, using default: np_meta.csv')

    with open(verbs_csv_path, 'w', newline='', encoding='utf-8') as verbs_file, \
         open(np_csv_path, 'w', newline='', encoding='utf-8') as np_file:
        
        verbs_writer = csv.DictWriter(verbs_file, fieldnames=['word', 'sentence_id', 'position_idx'])
        np_writer = csv.DictWriter(np_file, fieldnames=['word', 'sentence_id', 'position_idx'])
        
        verbs_writer.writeheader()
        np_writer.writeheader()
        
        for row in tqdm(df.itertuples(index=False), total=len(df)):
            labels, sentence, sentence_id = row.parsed_labels, row.parsed_sentence, row.sentence_id

            for idx, (label, part) in enumerate(zip(labels, sentence)):
                if not part:
                    continue

                prepared_row = {
                    'word': part,
                    'sentence_id': sentence_id,
                    'position_idx': idx
                }
                
                if label.startswith('V'):
                    verbs_writer.writerow(prepared_row)

                elif label.startswith('NN') or label == 'NP':
                    np_writer.writerow(prepared_row)
    
    return verbs_csv_path, np_csv_path
            

def embed_and_cluster(
    phrases: list[str], 
    pca_args: dict = None,
    threshold: float = 0.7
) -> list[list[int]]:
    '''
    Generates embeddings for input phrases, applies PCA dimensionality reduction,
    and performs agglomerative clustering based on cosine similarity with complete linkage 
    to group semantically similar phrases together.
    
    Args:
        phrases: List of text strings to embed and cluster.
        pca_args: Dictionary of parameters to pass to PCA initialization.
            Defaults to {'n_components': 50, 'svd_solver': 'full'}.
        threshold: Cosine similarity threshold for clustering (0-1).
            Higher values create tighter clusters. Defaults to 0.7.
    
    Returns:
        List of clusters, where each cluster is a list of phrase indices.
        Clusters are sorted by size in descending order.
    '''
    if pca_args is None:
        pca_args = {'n_components': 50, 'svd_solver': 'full'}

    vectors = EMBEDDING_MODEL.get_vectors(phrases, progress_bar=True)
    pca_model = PCA(**pca_args).fit(vectors)
    training_vectors = pca_model.transform(vectors)

    clust = AgglomerativeClustering(
        metric="cosine",
        linkage="complete",
        distance_threshold=1 - threshold,
        n_clusters=None
    )
    labels = clust.fit_predict(training_vectors)
    clusters = defaultdict(list)
    for i, lbl in enumerate(labels):
        clusters[lbl].append(i)

    clusters = list(clusters.values())
    clusters = sorted(clusters, key=lambda x: -len(x))

    return clusters


def create_clusters(
    input_path: str, save_folder_path: str, 
    pca_args: dict = None
):
    """
    Creates phrase clusters from a CSV file and save results.
    
    Reads phrases from a CSV with a 'word' column, groups them into clusters
    using embeddings and PCA, then saves clusters with their most frequent
    phrase as the label.
    
    Args:
        input_path: Path to input CSV file containing a 'word' column.
        save_folder_path: Directory path where clusters.csv will be saved.
        pca_args: Dictionary of PCA parameters. Defaults to 50 components
            with 'full' SVD solver.
    """
    if pca_args is None:
        pca_args = {'n_components': 50, 'svd_solver': 'full'}

    os.makedirs(save_folder_path, exist_ok=True)

    df = pd.read_csv(input_path)
    df['count'] = df.groupby('word')['word'].transform('count')
    phrase2count = df.drop_duplicates('word').set_index('word')['count'].to_dict()
    phrases = df['word'].unique().tolist()
    clusters = embed_and_cluster(phrases, pca_args)
    id2phrase = {idx: phrase for idx, phrase in enumerate(phrases)}
    
    with open(os.path.join(save_folder_path, 'clusters.csv'), 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'phrases', 'size'])
        for cluster in tqdm(clusters):
            phrases = [id2phrase[idx] for idx in cluster]
            counts = [phrase2count[ph] for ph in phrases]
            label = phrases[counts.index(max(counts))]
            size = sum(counts)
            writer.writerow([label, phrases, size])


def create_clusters_batched(
    input_path: str, 
    save_folder_path: str, 
    batch_size: int = 15000,
    pca_args: dict = None
):
    """
    Creates phrase clusters in batches for large datasets.

    Calls create_clusters() if dataset fits in one batch.
    
    Processes phrases in batches to handle memory constraints, creates
    initial clusters within batches, then performs second-level clustering
    on batch labels.
    
    Args:
        input_path: Path to input CSV file containing a 'word' column.
        save_folder_path: Directory path where output files will be saved.
        batch_size: Maximum number of phrases per batch. Defaults to 15000.
        pca_args: Dictionary of PCA parameters. Defaults to 50 components
            with 'full' SVD solver.
    
    """
    if pca_args is None:
        pca_args = {'n_components': 50, 'svd_solver': 'full'}

    os.makedirs(save_folder_path, exist_ok=True)

    pre_clusters_path = os.path.join(save_folder_path, 'pre_clusters.csv')
    clusters_path = os.path.join(save_folder_path, 'clusters.csv')

    df = pd.read_csv(input_path)
    df['count'] = df.groupby('word')['word'].transform('count')
    phrase2count = df.drop_duplicates('word').set_index('word')['count'].to_dict()
    phrases = df['word'].unique().tolist()
    if len(phrases) <= batch_size:
        return create_clusters(input_path, save_folder_path, pca_args)

    with open(pre_clusters_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'phrases', 'size'])
        for i in range(0, len(phrases), batch_size):
            batch = phrases[i:i+batch_size]
            print(f'Embedding batch {i//batch_size + 1} of {(len(phrases)-1)//batch_size + 1}')
            batch_clusters = embed_and_cluster(batch, pca_args)
            id2phrase = {idx: phrase for idx, phrase in enumerate(batch)}

            for cl in tqdm(batch_clusters): 
                cl_phrases = [id2phrase[idx] for idx in cl] 
                counts = [phrase2count[ph] for ph in cl_phrases] 
                cl_label = cl_phrases[counts.index(max(counts))] 
                cl_size = sum(counts)
                
                writer.writerow([
                    cl_label, cl_phrases, cl_size,
                ])

    df = pd.read_csv(pre_clusters_path, converters={'phrases': ast.literal_eval})
    label2phrases = {row.label: row.phrases for row in df.itertuples(index=False)}
    
    clusters = embed_and_cluster(df['label'].tolist(), pca_args) 
    with open(clusters_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'phrases', 'size'])
        for cluster in tqdm(clusters):
            df_subset = df.iloc[cluster]
            label2size = df_subset.groupby('label')['size'].sum().to_dict()
            label = max(label2size, key=label2size.get)
            size = sum(label2size.values())
            clustered_phrases = [
                phrase 
                for lab in df_subset['label']
                for phrase in label2phrases[lab]
            ]
            
            writer.writerow([label, clustered_phrases, size])


def replace_with_clusterized_labels(
    meta_path: str, 
    sentences_df: pd.DataFrame
) -> pd.DataFrame:
    """    
    Reads a metadata CSV containing phrase-to-label mappings and replaces
    matching phrases in the 'parsed_sentence' column with their cluster labels.
    
    Args:
        meta_path: Path to CSV file with 'phrases' and 'label' columns.
        sentences_df: DataFrame with 'parsed_sentence' column containing
                      lists of words/phrases.
    
    Returns:
        DataFrame with phrases replaced by cluster labels in 'parsed_sentence'.
    """
    meta_df = pd.read_csv(meta_path, converters={'phrases': ast.literal_eval})
    phrase2label = {
        phr.lower(): tup.label 
        for tup in meta_df.itertuples() 
        for phr in tup.phrases 
    }
    sentences_df = sentences_df.copy()
    sentences_df["parsed_sentence"] = sentences_df["parsed_sentence"].apply(
        lambda lst: [phrase2label.get(w.lower(), w) for w in lst]
    )

    return sentences_df


