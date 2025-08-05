import os
import json

import pandas as pd
from relatio import build_graph, draw_graph

from clusterize import get_roles_df, replace_roles_with_clusterized_labels

subfolder = 'sputnik_filtered_2020-2025_4+.csv'
folder = f'../results/relatio/{subfolder}'

roles_df = get_roles_df(folder)

for role in ['arg0', 'arg1', 'arg2']:
    resolved_path = os.path.join(folder, role, 'cluster_label_mapping_manual.csv')
    resolved_df = pd.read_csv(resolved_path, sep=";")
    resolved_df['cluster_indeces'] = resolved_df['cluster_indeces'].apply(lambda x: json.loads(x.replace("'", '"')))
    roles_df = replace_roles_with_clusterized_labels(role, resolved_df, roles_df, folder)


clean_roles_path = os.path.join(folder, 'clean_roles.csv')
roles_df.to_csv(clean_roles_path, index=False)

G = build_graph(
    roles_df, 
    top_n = 100, 
    prune_network = True
)

graph_path = os.path.join(folder, 'graph_100.html')
print('Drawing the graph')
draw_graph(
    G,
    notebook = True,
    show_buttons = False,
    width="1600px",
    height="1000px",
    output_filename = graph_path
)

filtered = roles_df[
    roles_df['ARG0'].str.contains("sweden|swedish", case=False, na=False) |
    roles_df['ARG1'].str.contains("sweden|swedish", case=False, na=False) |
    roles_df['ARG2'].str.contains("sweden|swedish", case=False, na=False)
]
#filtered_path = os.path.join(folder, 'filtered.csv')
#filtered.to_csv(filtered_path, index=False)

G = build_graph(
    filtered, 
    #top_n = 50, 
    top_n = 20, 
    prune_network = True
)

graph_path = os.path.join(folder, 'experiment.html')
#graph_path = os.path.join(folder, 'graph_50_sweden.html')

print('Drawing the graph')
draw_graph(
    G,
    notebook = True,
    show_buttons = False,
    width="1600px",
    height="1000px",
    output_filename = graph_path
)

G = build_graph(
    filtered, 
    top_n = 100, 
    prune_network = False
)

graph_path = os.path.join(folder, 'graph_100_sweden_not_pruned.html')
print('Drawing the graph')
draw_graph(
    G,
    notebook = True,
    show_buttons = False,
    width="1600px",
    height="1000px",
    output_filename = graph_path
)