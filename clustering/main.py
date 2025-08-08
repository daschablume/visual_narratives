import os

import pandas as pd

from clusterize import (
    replace_roles_with_clusterized_labels,
    load_lst_from_saved_txt,
    create_manual_clustering_with_batches
)
from create_graph import create_graph

from clusterize import (
    create_manual_clustering_with_batches
)

OUTPUT_FOLDER = f'../experiments'


def clusterize_srl(folder_path):
    '''
    Input: path to the folder with roles.csv file and path to the folder where to save the results.
    For now, assumes that the input file is called "roles.csv" and in the same directory as the output folders
    with roles.
    Also assumes that all the results are saved in the OUTPUT_FOLDER, which now is set to 'experiments' 
         <= one level up from the current directory
    '''
    inner_folder = os.path.join(OUTPUT_FOLDER, folder_path)
    roles_df = pd.read_csv(os.path.join(inner_folder, 'roles.csv'))

    for column in roles_df.columns:
        if 'ARG' not in column and 'V' not in column:
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing role: {column}")
        print('='*60)
        
        roles = roles_df.loc[roles_df[column].notna(), [column, 'sentence_id', 'image_id']]
        create_manual_clustering_with_batches(roles, column, inner_folder, 10000)


def update_roles_with_clusters(folder_path):
    '''
    Input: path to the folder with roles.csv file and path to the folder where to save the results.
    For now, assumes that the input file is called "roles.csv" and in the same directory as the output folders
    with roles.
    Also assumes that all the results are saved in the OUTPUT_FOLDER, which now is set to 'experiments' 
         <= one level up from the current directory
    '''
    # Update roles_df with clusterized roles
    inner_folder = os.path.join(OUTPUT_FOLDER, folder_path)
    updated_roles_df = pd.read_csv(f'{inner_folder}/roles.csv')
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
    
    updated_roles_df.to_csv(os.path.join(inner_folder, 'updated_roles.csv'), index=False)
    return updated_roles_df


