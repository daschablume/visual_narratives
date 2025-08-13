import os

import pandas as pd

from .clusterize import (
    replace_synt_parsed_with_clusterized_labels,
    prepare_synt_df_to_clustering,
    clusterize_synt_parsed
)
from utils import load_lst_from_saved_txt


def clusterize_srl(df: pd.DataFrame, output_dir: str, threshold=0.7, batch_size=20000):
    '''
    A wrapper function to clusterize roles from the roles.csv file.
    Iterates through each role column in the DataFrame and applies clustering to each role.
    Creates an inner folder `clusterized_roles` to store the results of clustering.
    Creates a separate folder for each role (inside `clusterized_roles`).

    Args:
        df (pd.DataFrame): DataFrame containing syntaxically parsed sentences.
        folder_path (str): Path to the folder containing the roles.csv file.
    '''
    verbs_meta, np_meta = prepare_synt_df_to_clustering(df)
    for phrase_type, phrases_meta in {'verbs_meta': verbs_meta, 'np_meta': np_meta}.items():
        inner_folder = os.path.join(output_dir, phrase_type)
        os.makedirs(inner_folder, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Clustering {phrase_type}...")
        clusterize_synt_parsed(
            phrases_meta, inner_folder,
            threshold=threshold, batch_size=batch_size
        )


def update_sentences_with_clusters(df, output_dir):
    '''
    A wrapper function to update sentences with clusterized labels.
    '''
    for phrase_type in ('verbs_meta', 'np_meta'):
        file_path = os.path.join(output_dir, phrase_type, 'clusters.csv')
        clusters_df = pd.read_csv(file_path, sep=';')
        clusters_df[['sentence_indices','image_indices', 'position_indices']] = load_lst_from_saved_txt(
            clusters_df, ['sentence_indices', 'image_indices', 'position_indices']
        )
        print(f"\n{'='*60}")
        print(f"Updating phrase_type: {phrase_type}")
    
        df = replace_synt_parsed_with_clusterized_labels(
            sentences_df=df, clusters_df=clusters_df)
    
    df.to_csv(os.path.join(output_dir, 'updated_roles.csv'), index=False)
    return df


