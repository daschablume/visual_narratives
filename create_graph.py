from collections import defaultdict
from copy import deepcopy
import glob
import json
import os
import random
import statistics

import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
from pyvis import network as net


NOT_NODE_LABELS = (
    'WHNP', 'TO', 'WHADVP', 'CC', 'RP', 'PRT', 'IN', 'RB', 'MD', 'DT'
)

# TODO: refactor this module, 
#       graph creation and graph analysis should be two different things

def create_graph(df: pd.DataFrame, only_verbs_labels=False):
    G = nx.MultiDiGraph()

    node2metadata = defaultdict(lambda: {'sentence_ids': set(), 'media_ids': set()})

    for _, row in df.iterrows():
        sentence_id = row['sentence_id']
        media_id = row['media_id']
        parsed_labels = row['parsed_labels']
        parsed_sentence = row['parsed_sentence']
        pointer1, pointer2 = 0, 1
        end_list = len(parsed_labels) - 1  # "." is always the last token

        while pointer2 < end_list:
            edge_label = ''
            # here I assume that the first label is always a noun phrase (NP)
            # and all the next times the first pointer points to the NP, since
            # pointer1 = pointer2 in the end of the loop
            
            while pointer2 < end_list:
                pos_tag = parsed_labels[pointer2]
                if pos_tag.startswith('V') or pos_tag == "MD":
                    # MD = modal verb
                    edge_label = parsed_sentence[pointer2]
                elif pos_tag.isalpha() and not parsed_labels[pointer2] in NOT_NODE_LABELS:
                    break
                else:
                    if not edge_label and pos_tag != "DT" and not only_verbs_labels:
                        # add edge label as a preposition only if it was not set by a verb
                        # and do not set it for a determiner (DT)
                        edge_label = parsed_sentence[pointer2]
                    elif pos_tag in ("RB", "MD") or pos_tag.startswith("V"):
                        # glue "not" and modals to a verb
                        edge_label += ' ' + parsed_sentence[pointer2]
                pointer2 += 1
            node1, node2  = parsed_sentence[pointer1], parsed_sentence[pointer2]
            pointer1 = pointer2
            pointer2 = pointer1 + 1
            # crotches =(
            if node1.lower() in ('it', 'they') or node2.lower() in ('it', 'they'):
                continue

            # check whether this edge already exists in the graph
            edge_found = False
            if G.has_edge(node1, node2):
                for _, edge_data in G[node1][node2].items():
                    if edge_data.get("label") == edge_label:
                        # if found and has the same label,
                        # increment the weight and add sentence_id and media_id
                        edge_data["weight"] += 1
                        edge_data["sentence_ids"].add(sentence_id)
                        edge_data["media_ids"].add(media_id)
                        edge_found = True
                        break

            # If no matching edge found, add new edge
            if not edge_found:
                G.add_edge(
                    node1,
                    node2,
                    label=edge_label,
                    weight=1,
                    sentence_ids={sentence_id},
                    media_ids={media_id},
                    pos=pos_tag
                )
            
            # Add sentence_id and media_id to both nodes' metadata
            for node in [node1, node2]:
                node2metadata[node]['sentence_ids'].add(sentence_id)
                node2metadata[node]['media_ids'].add(media_id)

    for node, metadata in node2metadata.items():
        if node in G:
            G.nodes[node]['sentence_ids'] = list(metadata['sentence_ids'])
            G.nodes[node]['media_ids'] = list(metadata['media_ids'])
    
    # convert sets to lists for serialization
    for _, _, _, data in G.edges(keys=True, data=True):
        if "sentence_ids" in data:
            data["sentence_ids"] = list(data["sentence_ids"])
        if "media_ids" in data:
            data["media_ids"] = list(data["media_ids"])
    
    return G


def draw_graph(
    networkx_graph,
    output_filename,
    notebook=True,
    width="1000px",
    height="1000px",
    show_buttons=False,
    only_physics_buttons=False,
):
    """
    This function accepts a networkx graph object,
    converts it to a pyvis network object preserving its node and edge attributes,
    and both returns and saves a dynamic network visualization.

    (For more info: https://pyvis.readthedocs.io/en/latest/documentation.html#pyvis.network.Network.add_node)

    Args:
        networkx_graph: The graph to convert and display
        notebook: Display in Jupyter?
        output_filename: Where to save the converted network
        width = width of the network
        height = height of th network
        show_buttons: Show buttons in saved version of network?
        only_physics_buttons: Show only buttons controlling physics of network?
    """

    # make a pyvis network
    pyvis_graph = net.Network(notebook=notebook, directed=True)
    pyvis_graph.repulsion()
    pyvis_graph.width = width
    pyvis_graph.height = height

    # Calculate weighted degree centrality and set node sizes
    degrees = dict(networkx_graph.degree(weight='weight'))
    max_degree = max(degrees.values())
    min_size = 15
    max_size = 100  # scaling leads to huge nodes
    for node in networkx_graph.nodes():
        normalized_size = min_size + (degrees[node] / max_degree) * (max_size - min_size)
        networkx_graph.nodes[node]['size'] = normalized_size

    betweenness = nx.betweenness_centrality(networkx_graph, weight='weight')
    # Normalize betweenness values to [0,1] for color mapping
    if betweenness:  # Check if not empty
        min_bet = min(betweenness.values())
        max_bet = max(betweenness.values())
        bet_range = max_bet - min_bet
        
        for node in networkx_graph.nodes():
            # Normalize to [0,1]
            if bet_range > 0:
                normalized_bet = (betweenness[node] - min_bet) / bet_range
            else:
                normalized_bet = 0
            
            # Map to blue→red color
            color = mcolors.to_hex((normalized_bet, 0, 1-normalized_bet))  # (R,G,B)
            networkx_graph.nodes[node]['color'] = color
    
    # Calculate edge thickness based on weight
    edge_weights = [edge_attrs['weight'] for _, _, edge_attrs in networkx_graph.edges(data=True)]
    min_weight = min(edge_weights)
    max_weight = max(edge_weights)
    weight_range = max_weight - min_weight

    min_width = 1
    max_width = 20  # Adjust these values as needed

    for source, target, edge_attrs in networkx_graph.edges(data=True):
        # Normalize weight to width range
        if weight_range > 0:
            normalized_width = min_width + (
                (edge_attrs['weight'] - min_weight) / weight_range) * (max_width - min_width)
        else:
            normalized_width = min_width
        
        edge_attrs['width'] = normalized_width

    for node, node_attrs in networkx_graph.nodes(data=True):
        pyvis_graph.add_node(node, **node_attrs)

    # for each edge and its attributes in the networkx graph
    for source, target, edge_attrs in networkx_graph.edges(data=True):
        pyvis_graph.add_edge(source, target, **edge_attrs)

    # turn buttons on
    if show_buttons:
        if only_physics_buttons:
            pyvis_graph.show_buttons(filter_=["physics"])
        else:
            pyvis_graph.show_buttons()

    # Make sure edges aren't written on one another
    pyvis_graph.set_edge_smooth("dynamic")

    # return and also save
    return pyvis_graph.show(output_filename)


def graph_with_strongest_edges(graph):
    '''
    Returns the graph with only the edges greater than the median of the edges weight.
    '''
    median_weight = statistics.median([
        d['weight'] for u, v, k, d in graph.edges(data=True, keys=True)
    ])
    filtered_edges = [
        (u, v, k) 
        for u, v, k, d in graph.edges(data=True, keys=True) 
        if d['weight'] > median_weight
    ]
    subgraph = graph.edge_subgraph(filtered_edges).copy()
    return subgraph


def print_narratives(node, graph):
    print('Narratives for node:', node)
    print('---')
    print('In edges')
    for source, target, data in graph.in_edges(node, data=True):
        edge_label = data.get('label', 'no_label') 
        edge_weight = data.get('weight', 1)
        print(f"{source} --{edge_label}({edge_weight})--> {target}")
    print('---')
    print('Out edges')
    for source, target, data in graph.out_edges(node, data=True):
        edge_label = data.get('label', 'no_label') 
        edge_weight = data.get('weight', 1)
        print(f"{source} --{edge_label}({edge_weight})--> {target}")


def make_subgraph_from_community(graph, community: set):
    community_graph = graph.subgraph(community).copy()
    return community_graph


def nodes_in_sentence(graph, sentence_id):
    '''Find all nodes that appear in a specific sentence'''
    nodes = []
    for node, data in graph.nodes(data=True):
        if sentence_id in data.get('sentence_ids', []):
            nodes.append(node)
    return nodes


def sentences_with_node(graph, node):
    '''Find all sentences containing a specific node'''
    return graph.nodes[node].get('sentence_ids', []) if node in graph else []


def save_graph_to_json(graph, path):
    data = nx.node_link_data(graph, edges="edges")
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def read_graph_from_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return nx.node_link_graph(data, edges="edges")


def analyze_graph(graph: nx.classes.multidigraph.MultiDiGraph, topn=10):
    '''
    A helper function to display betweenness centrality and degree centrality of the graph,
    as well as the number of nodes and edges.

    Eigenvector centrality is not calculated since nx doesn't support it for MultiDiGraph.
    '''
    #betweenness = nx.betweenness_centrality(graph, weight='weight')
    degree_centrality = nx.degree_centrality(graph)
    
    #top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:topn]
    top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:topn]
    
    #results = {
        #'betweenness': top_betweenness,
        #'degree': top_degree
    #}

    return top_degree


def dict_to_df(name, metrics_dict):
    '''
    Convert the output of analyze_graph to a pandas DataFrame
    '''
    frames = []
    for metric, values in metrics_dict.items():
        df = pd.DataFrame(values, columns=["term", metric])
        frames.append(df.set_index("term"))
    return pd.concat(frames, axis=1).reset_index().assign(source=name)


def print_strongest_connections(graph, threshold = 3):
    weights = [data['weight'] for u, v, data in graph.edges(data=True)]
    min_weight = min(weights)
    max_weight = max(weights)

    print(f"Min weight: {min_weight}")
    print(f"Max weight: {max_weight}")
    
    for u, v, data in graph.edges(data=True):
        if data['weight'] >= threshold:
            label = data.get('label', 'No label')
            print(f"Edge: {u} [{label}] {v}, Weight: {data['weight']}")


def clean_multigraph(graph, min_total_edge_weight=20, core=5, topn=50):
    # Calculate total weight between each node pair
    node_pair_weights = {}
    for u, v, k, d in graph.edges(keys=True, data=True):
        pair = (u, v)
        node_pair_weights[pair] = node_pair_weights.get(pair, 0) + d['weight']
    
    # Keep only edges between strongly connected node pairs
    strong_edges = []
    for u, v, k, d in graph.edges(keys=True, data=True):
        if node_pair_weights[(u, v)] >= min_total_edge_weight:
            strong_edges.append((u, v, k))
    
    G_filtered = nx.edge_subgraph(graph, strong_edges).copy()
    G_core = k_core_weighted_multigraph(G_filtered, k=core)
    degrees = dict(G_core.degree(weight='weight'))
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:topn]
    G_sub = G_core.subgraph(top_nodes).copy()
    return G_sub


def weighted_random_walk_no_cycles(graph, start_node, max_steps=5):
    """Weighted random walk - higher weight edges more likely to be chosen"""
    path = [start_node]
    current = start_node
    visited = {start_node}
    
    for step in range(max_steps):
        # Get unvisited neighbors with weights
        candidates = []
        weights = []
        
        for neighbor in graph[current]:
            if neighbor not in visited:
                if isinstance(graph, nx.MultiDiGraph):
                    # For MultiDiGraph, use maximum weight edge
                    max_weight = max(data['weight'] for data in graph[current][neighbor].values())
                else:
                    max_weight = graph[current][neighbor]['weight']
                candidates.append(neighbor)
                weights.append(max_weight)
        
        if not candidates:
            print(f"No unvisited neighbors. Stopping at step {step}")
            break
            
        # Weighted random choice
        next_node = random.choices(candidates, weights=weights)[0]
        chosen_weight = weights[candidates.index(next_node)]
        edge_label = dict(graph[current][next_node])[0]['label']
        print(f"Step {step+1}: {current} [{edge_label}] {next_node} (weight: {chosen_weight})")
        path.append(f'[{edge_label}]')
        path.append(next_node)
        visited.add(next_node)
        current = next_node
 
    return path


def k_core_weighted_multigraph(graph, k=2):
    new_graph = deepcopy(graph)
    changed = True
    while changed:
        changed = False
        # "weight" is supposed to sum the weights of parallel edges
        to_remove = [n for n in new_graph if new_graph.degree(n, weight='weight') < k]
        if to_remove:
            new_graph.remove_nodes_from(to_remove)
            changed = True
    return new_graph


def create_graph2(df: pd.DataFrame, only_verbs_labels=False):
    G = nx.MultiDiGraph()

    for _, row in df.iterrows():
        sentence_id = row['sentence_id']
        media_id = row['media_id']
        parsed_labels = row['parsed_labels']
        parsed_sentence = row['parsed_sentence']
        pointer1, pointer2 = 0, 1
        end_list = len(parsed_labels) - 1  # "." is always the last token

        while pointer2 < end_list:
            edge_label = ''
            # here I assume that the first label is always a noun phrase (NP)
            # and all the next times the first pointer points to the NP, since
            # pointer1 = pointer2 in the end of the loop
            
            while pointer2 < end_list:
                pos_tag = parsed_labels[pointer2]
                if pos_tag.startswith('V') or pos_tag == "MD":
                    # MD = modal verb
                    edge_label = parsed_sentence[pointer2]
                elif pos_tag.isalpha() and not parsed_labels[pointer2] in NOT_NODE_LABELS:
                    break
                else:
                    if not edge_label and pos_tag != "DT" and not only_verbs_labels:
                        # add edge label as a preposition only if it was not set by a verb
                        # and do not set it for a determiner (DT)
                        edge_label = parsed_sentence[pointer2]
                    elif pos_tag in ("RB", "MD") or pos_tag.startswith("V"):
                        # glue "not" and modals to a verb
                        edge_label += ' ' + parsed_sentence[pointer2]
                pointer2 += 1
            node1, node2  = parsed_sentence[pointer1], parsed_sentence[pointer2]
            pointer1 = pointer2
            pointer2 = pointer1 + 1

            # check whether this edge already exists in the graph
            edge_found = False
            if G.has_edge(node1, node2):
                for _, edge_data in G[node1][node2].items():
                    if edge_data.get("label") == edge_label:
                        # if found and has the same label,
                        # increment the weight and add sentence_id and media_id
                        edge_data["weight"] += 1
                        edge_found = True
                        break

            # If no matching edge found, add new edge
            if not edge_found:
                G.add_edge(
                    node1,
                    node2,
                    label=edge_label,
                    weight=1,
                    pos=pos_tag
                )

    return G


def split_df_create_graphs(merged_df, output_dir):
    '''
    Here: merged_df is an updated df together with event types.
    TODO: write a proper description, rename variables
    '''

    cop_c = merged_df[(merged_df['event'] == 'cop') & (merged_df['usr_type'] == 'c')]
    cop_m = merged_df[(merged_df['event'] == 'cop') & (merged_df['usr_type'] == 'm')]
    strike_c = merged_df[(merged_df['event'] == 'strike') & (merged_df['usr_type'] == 'c')]
    strike_m = merged_df[(merged_df['event'] == 'strike') & (merged_df['usr_type'] == 'm')]

    name2graph = {}
    for df in [cop_c, cop_m, strike_c, strike_m]:
        event_type, user_type = df['event'].iloc[0], df['usr_type'].iloc[0]
        print(f"Event: {event_type}, User type: {user_type}, Number of sentences: {len(df)}")
        print('Creating graph')
        graph = create_graph(df)
        print('Graph created with', len(graph.nodes), 'nodes and', len(graph.edges), 'edges.')
        #draw_graph(graph, output_filename=os.path.join(output_dir, f'graph_{event_type}_{user_type}.html'))
        save_graph_to_json(graph, path=os.path.join(output_dir, f'graph_{event_type}_{user_type}.json'))
        print('Graph saved to:', os.path.join(output_dir, f'graph_{event_type}_{user_type}.html'))
        name2graph[f'{event_type}_{user_type}'] = graph
    
    return name2graph


def analyze_graphs(folder: str=None, name2graph: dict=None):
    '''
    TODO: please write function description and proper ValueError
    '''
    if folder and name2graph:
        raise ValueError('')
    if not folder and not name2graph:
        raise ValueError
    if folder:
        graphs_path = glob(f'{folder}/*.json')
        name2graph = {}
        for path in graphs_path:
            name = path.split('/')[-1].split('.')[0]
            graph = read_graph_from_json(path)
            name2graph[name] = graph
    
    name2words = {}
    for name, graph in name2graph.items():
        results = analyze_graph(graph)
        words = [w for w, _ in results]
        name2words[name] = words
    
    for a, b, c, d in zip(
        name2words['cop_c'], name2words['cop_m'],
        name2words['strike_m'], name2words['strike_c']
    ):
        print(f"{a} & {b} & {c} & {d} \\\\")
    