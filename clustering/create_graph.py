from collections import defaultdict

import networkx as nx
import pandas as pd
from pyvis import network as net


def create_graph(df: pd.DataFrame):
    # TODO: check if I preserve the weights

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
    COPIED FROM RELATIO CODE: https://github.com/relatio-nlp/relatio

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

    # for each node and its attributes in the networkx graph
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

 