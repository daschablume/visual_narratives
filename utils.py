import ast
import json
import os
import re

import pandas as pd

from create_graph import create_graph, draw_graph, save_graph_to_json


def read_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    # Check if any row in Labels column needs JSON cleaning
    if df['Labels'].str.contains('json|```').any():
        df['Labels'] = df['Labels'].apply(_clean_json_string)
        # TODO2: 65 out of 1000 have different format (strings), clean later
        string_mask = df['Labels'].apply(lambda x: isinstance(x, str))
        return df[~string_mask].reset_index(drop=True)
    return df


def _clean_json_string(text):
    try:
        text = re.sub(r'^```json\s*', '', text)  # Remove ```json at start
        text = re.sub(r'^json\s*', '', text)     # Remove json at start
        text = re.sub(r'```$', '', text)         # Remove ``` at end
        text = text.replace("'", '"')            # Replace single quotes with double quotes
        return json.loads(text)
    except json.JSONDecodeError:
        return text
 

def check_all_unique_pos_tags(df):
    tags = df['parsed_labels'].apply(lambda x: set(x)).explode().unique()
    return tags


def read_updated_roles(prompt_num, full=False, saved_dir='../experiments3_better_graphs'):
    full_piece = 'full' if full else '300sent'
    file_path = f'prompt{prompt_num}_synt_pars_{full_piece}/updated_roles.csv'
    df = pd.read_csv(os.path.join(saved_dir, file_path), converters={'parsed_labels': ast.literal_eval, 'parsed_sentence': ast.literal_eval})
    return df


def create_draw_graph_from_saved(prompt_num, graph_name, full=False, saved_dir='../experiments3_better_graphs'):
    full_piece = 'full' if full else '300sent'
    output_dir = os.path.join(saved_dir, f'prompt{prompt_num}_synt_pars_{full_piece}')
    df = read_updated_roles(prompt_num, full=full, saved_dir=saved_dir)
    graph = create_graph(df)
    if not graph_name.endswith('.html'):
        graph_name += '.html'
    draw_graph(graph, output_filename=os.path.join(output_dir, graph_name))
    print('Graph drawn and saved to:', os.path.join(output_dir, graph_name))
    save_graph_to_json(graph, path=os.path.join(output_dir, 'graph.json'))
    return graph


def read_climate_movements(folder='../climate_movements'):
    df1 = pd.read_csv(os.path.join(folder, 'narr_strike_e.tsv'), sep='\t')
    df2 = pd.read_csv(os.path.join(folder, 'narr_strike_o.tsv'), sep='\t')
    return pd.concat([df1, df2], ignore_index=True)