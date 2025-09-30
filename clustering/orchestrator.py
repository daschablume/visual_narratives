import ast
import os

import pandas as pd

from .clusterize import (
    prepare_df_to_clustering,
    create_clusters_batched,
    replace_with_clusterized_labels,
    replace_with_lemmatized_verbs
)


def process_and_cluster_phrases(df: pd.DataFrame, output_dir: str, threshold=0.85, batch_size=10000):
    '''
    A wrapper function to clusterize roles from the roles.csv file.
    Iterates through each role column in the DataFrame and applies clustering to each role.
    Creates an inner folder `clusterized_roles` to store the results of clustering.
    Creates a separate folder for each role (inside `clusterized_roles`).

    Args:
        df (pd.DataFrame): DataFrame containing syntaxically parsed sentences.
        folder_path (str): Path to the folder containing the roles.csv file.
    '''
    verbs_path, nouns_path = os.path.join(output_dir, 'verbs.csv'), os.path.join(output_dir, 'np.csv')
    print('Preparing data for clustering...')
    _, np_phrases_path = prepare_df_to_clustering(df, verbs_csv_path=verbs_path, np_csv_path=nouns_path)
    np_out_folder = os.path.join(output_dir, 'np_meta')
    os.makedirs(np_out_folder, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Clustering np_meta...")
    create_clusters_batched(
        np_phrases_path, np_out_folder,
        threshold=threshold, batch_size=batch_size
    )


def update_sentences_with_clusterized(df, output_dir):
    '''
    A wrapper function to update sentences with clusterized labels.
    '''
    print(f"Updating noun phrases with clusterized labels...")
    np_path = os.path.join(output_dir, 'np_meta', 'clusters.csv')
    clusters_df = pd.read_csv(
        np_path, sep=';',
        converters={
            'sentence_indices': ast.literal_eval, 'position_indices': ast.literal_eval}
    )
    df = replace_with_clusterized_labels(sentences_df=df, clusters_df=clusters_df)
    print(f"\n{'='*60}")
    
    print(f'Updating with lemmatized verbs')
    verbs_path = os.path.join(output_dir, 'verbs.csv')
    lemmas_df = pd.read_csv(verbs_path)
    df = replace_with_lemmatized_verbs(lemmas_df, df)
    df.to_csv(os.path.join(output_dir, 'updated_roles.csv'), index=False)
    return df


