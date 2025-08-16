import os

import pandas as pd

from .clusterize import (
    create_clusters,
    replace_roles_with_clusterized_labels,
    
)
from utils import load_lst_from_saved_txt


def clusterize_srl(folder_path: str):
    '''
    A wrapper function to clusterize roles from the roles.csv file.
    Iterates through each role column in the DataFrame and applies clustering to each role.
    Creates an inner folder `clusterized_roles` to store the results of clustering.
    Creates a separate folder for each role (inside `clusterized_roles`).

    Args:
        folder_path (str): Path to the folder containing the roles.csv file.
    '''
    roles_df = pd.read_csv(os.path.join(folder_path, 'roles.csv'))
    inner_folder = os.path.join(folder_path, 'clusterized_roles')
    os.makedirs(inner_folder, exist_ok=True)

    for column in roles_df.columns:
        if 'ARG' not in column and 'V' not in column:
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing role: {column}")
        print('='*60)
        
        roles = roles_df.loc[roles_df[column].notna(), [column, 'sentence_id', 'image_id']]
        create_clusters(roles, column, inner_folder, 10000)


def update_roles_with_clusters(folder_path):
    '''
    A wrapper function to update roles with clusterized labels.
    Reads the roles.csv file and replaces each role with its corresponding clusterized label.
    For clusterized labels, iterates over each folder inside `clusterized_roles` 
        and updates the roles DataFrame.
    
    Args:
        folder_path (str): Path to the folder containing the `roles.csv` file and `clusterized_roles/`.

    Returns:
        pd.DataFrame: Updated roles DataFrame with clusterized labels.
    '''
    updated_roles_df = pd.read_csv(f'{folder_path}/roles.csv')
    inner_folder = os.path.join(folder_path, 'clusterized_roles')
    updated_roles_df[['sentence_lst', 'ordered_roles']] = load_lst_from_saved_txt(
        updated_roles_df, ['sentence_lst', 'ordered_roles'])
    for column in updated_roles_df.columns:
        if 'ARG' not in column and 'V' not in column:
            continue
        
        print(f"\n{'='*60}")
        print(f"Updating role: {column}")
        resolved_path = os.path.join(inner_folder, column, 'cluster_label_mapping.csv')
        resolved_df = pd.read_csv(resolved_path, sep=";")
        resolved_df[['image_indices', 'sentence_indices']] = load_lst_from_saved_txt(
            resolved_df, ['image_indices', 'sentence_indices'])

        updated_roles_df = replace_roles_with_clusterized_labels(column, resolved_df, updated_roles_df)
    
    updated_roles_df.to_csv(os.path.join(folder_path, 'updated_roles.csv'), index=False)
    return updated_roles_df


