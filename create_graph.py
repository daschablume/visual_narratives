from collections import defaultdict
from copy import deepcopy
import json
import random

import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
from pyvis import network as net

NOT_NODE_LABELS = (
    'WHNP', 'TO', 'WHADVP', 'CC', 'RP', 'PRT', 'IN', 'RB', 'MD', 'DT'
)


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
    
    # covert sets to lists for serialization
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
    degree_range = max_degree - min(degrees.values())
    min_size = 15
    max_size = min_size + (degree_range * 0.5)
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
    max_width = 5  # Adjust these values as needed

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
    betweenness = nx.betweenness_centrality(graph, weight='weight')
    degree_centrality = nx.degree_centrality(graph)
    closeness_centrality = nx.closeness_centrality(graph, distance='weight')
    page_rank = nx.pagerank(graph, weight='weight')
    
    # Get top 20 nodes for each metric
    top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:topn]
    top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:topn]
    top_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:topn]
    top_pagerank = sorted(page_rank.items(), key=lambda x: x[1], reverse=True)[:topn]
    
    results = {
        'betweenness': top_betweenness,
        'degree': top_degree,
        'closeness': top_closeness,
        'pagerank': top_pagerank
    }

    return results


def dict_to_df(name, metrics_dict):
    '''
    Convert the output of analyze_graph to a pandas DataFrame
    '''
    frames = []
    for metric, values in metrics_dict.items():
        df = pd.DataFrame(values, columns=["term", metric])
        frames.append(df.set_index("term"))
    return pd.concat(frames, axis=1).reset_index().assign(source=name)


def compare_basic_stats(G1, G2, name1="Graph1", name2="Graph2"):
    stats = {
        'Nodes': (G1.number_of_nodes(), G2.number_of_nodes()),
        'Edges': (G1.number_of_edges(), G2.number_of_edges()),
        'Density': (nx.density(G1), nx.density(G2)),
        'Avg Degree': (sum(dict(G1.degree()).values())/G1.number_of_nodes(), 
                      sum(dict(G2.degree()).values())/G2.number_of_nodes())
    }
    
    for metric, (v1, v2) in stats.items():
        print(f"{metric}: {name1}={v1:.3f}, {name2}={v2:.3f}")

    nodes1, nodes2 = set(G1.nodes()), set(G2.nodes())
    common = nodes1 & nodes2
    only1 = nodes1 - nodes2
    only2 = nodes2 - nodes1

    print(f"Common nodes: {len(common)}/{len(nodes1 | nodes2)} total")
    print(f"Only in {name1}: {len(only1)}, Only in {name2}: {len(only2)}")

    deg1 = dict(G1.degree())
    deg2 = dict(G2.degree())
    top1 = sorted(deg1, key=deg1.get, reverse=True)[:10]
    top2 = sorted(deg2, key=deg2.get, reverse=True)[:10]

    print(f"Top nodes {name1}", top1)
    print(f"Top nodes {name2}", top2)
    print("Overlap in top 10:", len(set(top1) & set(top2)))

    weights1 = [d['weight'] for _, _, d in G1.edges(data=True)]
    weights2 = [d['weight'] for _, _, d in G2.edges(data=True)]

    print(f"Avg edge weight: {name1}={np.mean(weights1):.2f}, {name2}={np.mean(weights2):.2f}")
    print(f"Max edge weight: {name1}={max(weights1)}, {name2}={max(weights2)}")

    # Top 10 overlapping nodes by degree
    print(f"\nTop 10 overlapping nodes:")
    common_degrees = [(node, deg1.get(node, 0), deg2.get(node, 0)) for node in common]
    top_common = sorted(common_degrees, key=lambda x: x[1] + x[2], reverse=True)[:10]
    for node, d1, d2 in top_common:
        print(f"  {node}: {name1}={d1}, {name2}={d2}")

    # Top 10 non-overlapping nodes
    print(f"\nTop 10 non-overlapping nodes from {name1}:")
    top_only1 = sorted([(n, deg1[n]) for n in only1], key=lambda x: x[1], reverse=True)[:10]
    for node, degree in top_only1:
        print(f"  {node}: degree={degree}")
        
    print(f"\nTop 10 non-overlapping nodes from {name2}:")
    top_only2 = sorted([(n, deg2[n]) for n in only2], key=lambda x: x[1], reverse=True)[:10]
    for node, degree in top_only2:
        print(f"  {node}: degree={degree}")


def compare_betweenness_triplets(G1, G2, name1="Graph1", name2="Graph2", top_n=20):
    def get_top_betweenness_triplets(G, n=20):
        # Calculate betweenness centrality (this is the expensive part)
        print(f"Calculating betweenness centrality for {G.number_of_nodes()} nodes...")
        betweenness = nx.betweenness_centrality(G, k=min(1000, G.number_of_nodes()))  # Sample for large graphs
        
        # Get top nodes by betweenness
        top_bet_nodes = sorted(betweenness, key=betweenness.get, reverse=True)[:50]  # Get more candidates
        
        triplets = []
        for node in top_bet_nodes:
            # Find heaviest edge from this node
            max_weight = 0
            best_edge = None
            
            for target, edges in G[node].items():
                # Sum weights for all edges to this target (multigraph)
                total_weight = sum(edge_data['weight'] for edge_data in edges.values())
                if total_weight > max_weight:
                    max_weight = total_weight
                    # Get the edge with highest individual weight for the label
                    best_individual_edge = max(edges.values(), key=lambda x: x['weight'])
                    best_edge = (target, total_weight, best_individual_edge.get('label', ''))
            
            if best_edge:
                triplets.append((node, betweenness[node], best_edge[0], best_edge[1], best_edge[2]))
        
        # Sort by betweenness and return top n
        triplets.sort(key=lambda x: x[1], reverse=True)
        return triplets[:n]
    
    # Get triplets from both graphs
    triplets1 = get_top_betweenness_triplets(G1, top_n)
    triplets2 = get_top_betweenness_triplets(G2, top_n)
    
    print(f"\nTop {top_n} Betweenness Centrality Triplets - {name1}:")
    for i, (node, bet, target, weight, label) in enumerate(triplets1, 1):
        print(f"{i:2d}. {node} <{label}({weight})> {target} (betweenness: {bet:.4f})")
    
    print(f"\nTop {top_n} Betweenness Centrality Triplets - {name2}:")
    for i, (node, bet, target, weight, label) in enumerate(triplets2, 1):
        print(f"{i:2d}. {node} <{label}({weight})> {target} (betweenness: {bet:.4f})")
    
    # Compare overlaps
    nodes1 = set(t[0] for t in triplets1)
    nodes2 = set(t[0] for t in triplets2)
    common_nodes = nodes1 & nodes2
    
    print(f"\nComparison:")
    print(f"Common high-betweenness nodes: {len(common_nodes)}/{len(nodes1 | nodes2)}")
    if common_nodes:
        print(f"Common nodes: {list(common_nodes)}")


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