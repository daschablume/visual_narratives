import os

import pandas as pd

from .clusterize import (
    prepare_df_to_clustering,
    create_clusters_batched,
    replace_with_clusterized_labels,
)


def process_and_cluster_phrases(
    df: pd.DataFrame, output_dir: str, 
    pca_args: dict = None,
    batch_size=15000
):
    '''
    A wrapper function to clusterize roles from the roles.csv file.
    Iterates through each role column in the DataFrame and applies clustering to each role.
    Creates an inner folder `clusterized_roles` to store the results of clustering.
    Creates a separate folder for each role (inside `clusterized_roles`).

    Args:
        df (pd.DataFrame): DataFrame containing syntaxically parsed sentences.
        folder_path (str): Path to the folder containing the roles.csv file.
    '''
    if pca_args is None:
        pca_args = {'n_components': 50, 'svd_solver': 'full'}

    verbs_path, np_phrases_path = os.path.join(output_dir, 'verbs.csv'), os.path.join(output_dir, 'np.csv')
    print('Preparing data for clustering...')
    prepare_df_to_clustering(df, verbs_csv_path=verbs_path, np_csv_path=np_phrases_path)
    for meta, path in {'verbs_meta': verbs_path, 'nouns_meta': np_phrases_path}.items():
        out_folder = os.path.join(output_dir, meta)
        os.makedirs(out_folder, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Clustering {meta}...")
        create_clusters_batched(
            path, out_folder, pca_args=pca_args,
            batch_size=batch_size
        )


def update_sentences_with_clusterized(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    '''
    A wrapper function to update sentences with clusterized labels.
    '''
    meta = {"nouns_meta": 'noun_phrases', "verbs_meta": 'verbs'}
    for folder_name, phrase_type in meta.items():
        print(f"Updating {phrase_type} phrases with clusterized labels...")
        meta_path = os.path.join(output_dir, folder_name, 'clusters.csv')
        df = replace_with_clusterized_labels(meta_path, df)
    
    return df


