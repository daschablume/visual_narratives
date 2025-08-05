import os
import json

import pandas as pd

from clusterize import get_roles_df, replace_roles_with_clusterized_labels

from clusterize import (
    create_manual_clustering_with_batches
)

subfolder = 'prompt_4_all_roles'
folder = f'../experiments/{subfolder}'

roles_df = pd.read_csv(f'{folder}/roles.csv')

for column in roles_df.columns:
    if 'ARG' not in column:
        continue
    
    print(f"\n{'='*60}")
    print(f"Processing role: {column}")
    print('='*60)
    
    roles = roles_df.loc[roles_df[column].notna(), column].tolist()
    create_manual_clustering_with_batches(roles, column, folder, 10000)


roles_df = pd.read_csv(f'{folder}/roles.csv')
for column in roles_df.columns:
    if 'ARG' not in column:
        continue
    resolved_df['cluster_indices'] = resolved_df['cluster_indices'].apply(lambda x: json.loads(x.replace("'", '"')))
    try:
        roles_df = replace_roles_with_clusterized_labels(column, resolved_df, roles_df, folder)
    except:
        print(f"Error processing column {column}. Skipping...")
        continue
