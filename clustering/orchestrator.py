import ast
import os

import pandas as pd

from .clusterize import (
    prepare_df_to_clustering,
    create_clusters_batched,
    replace_with_clusterized_labels,
)


def process_and_cluster_phrases(
    df: pd.DataFrame, output_dir: str, 
    pca_args: dict = {'n_components': 50, 'svd_solver': 'full'},
    batch_size=10000
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


def update_sentences_with_clusterized(df, output_dir):
    '''
    A wrapper function to update sentences with clusterized labels.
    '''
    meta = {"np_meta": 'noun', "verbs_meta": 'verb'}
    for folder_name, phrase_type in meta.items():
        print(f"Updating {phrase_type} phrases with clusterized labels...")
        meta_path = os.path.join(output_dir, folder_name, 'clusters.csv')
        df = replace_with_clusterized_labels(meta_path, df)
    
    return df


