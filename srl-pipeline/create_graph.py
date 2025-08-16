from collections import defaultdict

import networkx as nx
import pandas as pd


def create_graph(df: pd.DataFrame):
    '''
    Create graph from a DataFrame containing semantic role labels.
    Warning! 
    The weights are not preserved. This function works but probably should be a bit updated.
    '''
    G = nx.MultiDiGraph()

    node2metadata = defaultdict(lambda: {'sentence_ids': set(), 'image_ids': set()})

    for _, row in df.iterrows():
        sentence_id = row['sentence_id']
        image_id = row['image_id']
        roles_order = row['ordered_roles']
        pointer1, pointer2 = 0, 1
        end_list = len(roles_order)
        
        while pointer2 < end_list:
            label = ''
            role1, role2 = roles_order[pointer1], roles_order[pointer2]
            
            if role2 == 'V':
                pointer2 += 1
                if pointer2 >= end_list:
                    break
                label = row[role2]
                pointer1 += 1  # jump over the verb
                role2 = roles_order[pointer2]
            
            node1, node2 = row[role1], row[role2]
            
            # Add sentence_id and image_id to both nodes' metadata
            for node in [node1, node2]:
                node2metadata[node]['sentence_ids'].add(sentence_id)
                node2metadata[node]['image_ids'].add(image_id)

            G.add_edge(
                node1,
                node2,
                label=label,
                hidden=False,
                sentence_id=sentence_id,  # Edge-level metadata
                image_id=image_id
            )

            pointer1 += 1
            pointer2 += 1

    for node, metadata in node2metadata.items():
        if node in G:
            G.nodes[node]['sentence_ids'] = list(metadata['sentence_ids'])
            G.nodes[node]['image_ids'] = list(metadata['image_ids'])
    
    return G
