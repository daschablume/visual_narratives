from collections import defaultdict
import json

import matplotlib.colors as mcolors
import networkx as nx
import pandas as pd
from pyvis import network as net


def create_graph(df: pd.DataFrame):
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


def create_graph_srl(df: pd.DataFrame):
    G = nx.MultiDiGraph()

    # TODO1: still save punctuation for some cases

    node2metadata = defaultdict(lambda: {'sentence_ids': set(), 'image_ids': set()})

    for row_id, row in df.iterrows():
        sentence_id = row['sentence_id']
        image_id = row['id']
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
                if pos_tag.startswith('V'):
                    edge_label = parsed_sentence[pointer2]
                elif pos_tag.isalpha() and not parsed_labels[pointer2] in ('IN', 'CC', 'RB'):
                    break
                else:
                    if not edge_label and pos_tag.isalpha():
                        # add edge label as a preposition only if it was not set by a verb
                        # and do not set it for punctuation
                        edge_label = parsed_sentence[pointer2]
                    elif edge_label and pos_tag == "RB":
                        # add "not", if it is a negation
                        edge_label += ' ' + parsed_sentence[pointer2]
                pointer2 += 1
            node1, node2  = parsed_sentence[pointer1], parsed_sentence[pointer2]
            if node1 == '.' or node2 == '.':
                print(f"FOUND PUNCTUATION: {node1}, {node2} in row_id: {row_id}")
            pointer1 = pointer2
            pointer2 = pointer1 + 1

            # check whether this edge already exists in the graph
            edge_found = False
            if G.has_edge(node1, node2):
                for _, edge_data in G[node1][node2].items():
                    if edge_data.get("label") == edge_label:
                        # if found and has the same label,
                        # increment the weight and add sentence_id and image_id
                        edge_data["weight"] += 1
                        edge_data["sentence_ids"].add(sentence_id)
                        edge_data["image_ids"].add(image_id)
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
                    image_ids={image_id}
                )
            
            # Add sentence_id and image_id to both nodes' metadata
            for node in [node1, node2]:
                node2metadata[node]['sentence_ids'].add(sentence_id)
                node2metadata[node]['image_ids'].add(image_id)

    for node, metadata in node2metadata.items():
        if node in G:
            G.nodes[node]['sentence_ids'] = list(metadata['sentence_ids'])
            G.nodes[node]['image_ids'] = list(metadata['image_ids'])
    
    # covert sets to lists for serialization
    for _, _, _, data in G.edges(keys=True, data=True):
        if "sentence_ids" in data:
            data["sentence_ids"] = list(data["sentence_ids"])
        if "image_ids" in data:
            data["image_ids"] = list(data["image_ids"])
    
    return G


def draw_graph(
    networkx_graph,
    notebook=True,
    output_filename="graph.html",
    width="1000px",
    height="1000px",
    show_buttons=False,
    only_physics_buttons=False,
):
    """
    This function accepts a networkx graph object,
    converts it to a pyvis network object preserving its node and edge attributes,
    and both returns and saves a dynamic network visualization.

    Valid node attributes include:
        "size", "value", "title", "x", "y", "label", "color".

        (For more info: https://pyvis.readthedocs.io/en/latest/documentation.html#pyvis.network.Network.add_node)

    Valid edge attributes include:
        "arrowStrikethrough", "hidden", "physics", "title", "value", "width"

        (For more info: https://pyvis.readthedocs.io/en/latest/documentation.html#pyvis.network.Network.add_edge)


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
    pyvis_graph.width = width
    pyvis_graph.height = height

    # Calculate weighted degree centrality and set node sizes
    degrees = dict(networkx_graph.degree(weight='weight'))
    max_degree = max(degrees.values())
    degree_range = max_degree - min(degrees.values())
    min_size = 15
    max_size = min_size + (degree_range * 0.5)

    for node in networkx_graph.nodes():
        # Normalize degree to size range
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
    