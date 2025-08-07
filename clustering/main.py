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

subfolder = 'prompt_4_all_roles'
folder = f'../experiments/{subfolder}'

roles_df = pd.read_csv(f'{folder}/roles.csv')

for column in roles_df.columns:
    if 'ARG' not in column and 'V' not in column:
        continue
    
    print(f"\n{'='*60}")
    print(f"Processing role: {column}")
    print('='*60)
    
    roles = roles_df.loc[roles_df[column].notna(), [column, 'sentence_id', 'image_id']]
    create_manual_clustering_with_batches(roles, column, folder, 10000)

# Update roles_df with clusterized roles
updated_roles_df = pd.read_csv(f'{folder}/roles.csv')
updated_roles_df[['sentence_lst', 'ordered_roles']] = load_lst_from_saved_txt(
    updated_roles_df, ['sentence_lst', 'ordered_roles'])
for column in updated_roles_df.columns:
    if 'ARG' not in column and 'V' not in column:
        continue
    
    print(f"\n{'='*60}")
    print(f"Updating role: {column}")
    resolved_path = os.path.join(folder, column, 'cluster_label_mapping.csv')
    resolved_df = pd.read_csv(resolved_path, sep=";")
    resolved_df[['image_indices', 'sentence_indices']] = load_lst_from_saved_txt(
        resolved_df, ['image_indices', 'sentence_indices'])

    updated_roles_df = replace_roles_with_clusterized_labels(column, resolved_df, updated_roles_df)

# Create graph
graph = create_graph(updated_roles_df)
