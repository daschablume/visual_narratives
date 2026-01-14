import ast
from copy import deepcopy
import json
import os

import networkx as nx
from networkx.algorithms.community import louvain_communities
import pandas as pd

from .draw_graph import draw_graph
from .build_graph import create_graph

COP, STRIKE = 'cop', 'strike'
MOVEMENT, COUNTERMOVEMENT = 'm', 'c'


def extract_communities(graph: nx.Graph, seed=42) -> list[set]:
    '''
    Detect communities in a graph using the Louvain method.
    Returns a list of sets, where each set contains the nodes belonging to a community.
    '''
    return louvain_communities(graph, weight='weight', seed=seed)


def make_subgraph_from_community(graph: nx.Graph, community: set) -> nx.Graph:
    """Create a subgraph containing only nodes from the specified community."""
    community_graph = graph.subgraph(community).copy()
    return community_graph


def save_graph_to_json(graph: nx.Graph, path: str):
    data = nx.node_link_data(graph, edges="edges")
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def read_graph_from_json(path: str) -> nx.Graph:
    with open(path, 'r') as f:
        data = json.load(f)
    return nx.node_link_graph(data, edges="edges")


def k_core_weighted_multigraph(graph: nx.Graph, k: int=2) -> nx.Graph:
    '''
    Extract the k-core of a weighted multigraph by iteratively removing nodes
    with weighted degree less than k.
    '''
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


def ego_without_ego(graph: nx.Graph, center_node: str) -> nx.Graph:
    '''
    Create an ego graph centered on a specified node, excluding the center node itself.
    '''
    ego_graph = nx.ego_graph(graph, center_node, radius=1)
    ego_graph.remove_node(center_node)
    return ego_graph


def make_and_draw_ego_without_ego(
    graph: nx.Graph,
    center_node: str,
    event: str,
    side: str,
    output_dir: str
) -> nx.Graph:  
    """
    Create an ego graph without the center node and save visualization to HTML.
    
    Args:
        graph: The source graph object to extract the ego network from.
        center_node: The central node for the ego network extraction.
        side: Side identifier for filename (e.g., 'm' for movement and 'c' for countermovement).
        event: Event identifier for filename ('cop' or 'strike').
        output_dir: Directory path where the HTML file will be saved.
    
    Returns:
        The generated ego graph (without center node).
    
    """
    ego_graph = ego_without_ego(graph, center_node)
    filename = f'{center_node}_{event}_{side}_ego_no_ego.html'
    output_path = os.path.join(output_dir, filename)
    draw_graph(
        ego_graph,
        output_filename=output_path,
    )
    return ego_graph


def make_and_draw_multiple_egoless(
    name2graph: dict[str, nx.Graph], ego: str, output_dir: str
) -> dict[str, nx.Graph]:
    """
    Create and draw ego-less versions of multiple graphs by removing a specified ego node.
    
    Args:
        name2graph: Dictionary mapping graph names to NetworkX Graph objects. 
                   Graph names are expected to follow the format "event_side".
        ego: The identifier of the ego node to remove from each graph.
        output_dir: Directory path where the ego-less graph visualizations will be saved.
    
    Returns:
        Dictionary mapping graph names to their ego-less versions (NetworkX Graph objects).
    """
    name2egoless = {}
    for name, graph in name2graph.items():
        event, side = name.split('_')  # for example, "cop_m"
        try:
            egoless_graph = make_and_draw_ego_without_ego(graph, ego, event, side, output_dir)
            name2egoless[name] = egoless_graph
            print(f"Drew ego-less graph for {name}")
        except Exception as e:
            print(f"Could not draw ego-less graph for {name}: {e}")
    return name2egoless


def make_and_draw_ego_graph(
    graph: nx.Graph, center_node: str, org: str, side: str, output_dir: str, 
    toggle_physics=True, undirected=True
) -> nx.Graph:
    '''
    Creates an ego graph centered on a specified node and saves it as HTML visualization and JSON.
    '''
    if center_node not in graph:
        raise ValueError(f"Center node '{center_node}' not found in graph. "
                        f"Available nodes: {list(graph.nodes())[:10]}...")
    ego_graph = nx.ego_graph(graph, center_node, radius=1, undirected=undirected)
    output_path = os.path.join(output_dir, f'{center_node}_{org}_{side}_ego.html')
    draw_graph(
        ego_graph,
        output_filename=output_path,
        toggle_physics=toggle_physics
    )
    print('Ego graph drawn to:', output_path)

    output_json_path = os.path.join(output_dir, f'{center_node}_{org}_{side}_ego.json')
    save_graph_to_json(ego_graph, path=output_json_path)
    print('Ego graph saved to:', output_json_path)
    return ego_graph


def make_and_draw_ego_graphs(
    name2graph: dict[str, nx.Graph], center_node: str, output_dir: str, 
    toggle_physics: bool=True, undirected: bool=True
) -> dict[str, nx.Graph]:
    '''
    Creates and draws ego graphs for multiple named graphs centered on a specific node.
    '''
    name2ego = {}
    for name, graph in name2graph.items():
        org, side = name.split('_')
        try:
            ego_graph = make_and_draw_ego_graph(
                graph, center_node, org, output_dir, side, 
                toggle_physics=toggle_physics, undirected=undirected
            )
            name2ego[name] = ego_graph
        except Exception as e:
            print(f"Could not draw ego graph for {name}: {e}")
    
    return name2ego


def split_df_create_graphs(
    merged_df: pd.DataFrame, output_dir:str, 
    only_verbs_labels:bool=True, draw_graphs:bool=False
) -> dict:
    '''
    Splits a merged dataframe by event type and user type combinations, then creates 
    and saves graph representations for each combination.
    
    Args:
        merged_df (pd.DataFrame): DataFrame containing columns 'event' (event type), 
                                   'usr_type' (user type), and other data needed for 
                                   graph creation
        output_dir (str): Directory path where graph JSON files will be saved
        only_verbs_labels (bool): Flag passed to create_graph() to control labeling 
                                  behavior. Defaults to True
        draw_graphs (bool): Whether to draw and save the graphs as HTML files.
                            Defaults to False (can be super slow for big datasets).
    
    Returns:
        dict: Dictionary mapping '{event_type}_{user_type}' strings to their 
              corresponding graph objects
    
    Expected event types: 'cop', 'strike'
    Expected user types: 'c', 'm'
    '''
    cop_c = merged_df[(merged_df['event'] == COP) & (merged_df['usr_type'] == COUNTERMOVEMENT)]
    cop_m = merged_df[(merged_df['event'] == COP) & (merged_df['usr_type'] == MOVEMENT)]
    strike_c = merged_df[(merged_df['event'] == STRIKE) & (merged_df['usr_type'] == COUNTERMOVEMENT)]
    strike_m = merged_df[(merged_df['event'] == STRIKE) & (merged_df['usr_type'] == MOVEMENT)]

    name2graph = {}
    for df in [cop_c, cop_m, strike_c, strike_m]:
        if df.empty:
            print("Warning: Empty dataframe found, skipping...")
            continue
        event_type, user_type = df['event'].iloc[0], df['usr_type'].iloc[0]
        print(f"Event: {event_type}, User type: {user_type}, Number of sentences: {len(df)}")
        print('Creating graph')
        graph = create_graph(df, only_verbs_labels=only_verbs_labels)
        print('Graph created with', len(graph.nodes), 'nodes and', len(graph.edges), 'edges.')
        save_graph_to_json(graph, path=os.path.join(output_dir, f'graph_{event_type}_{user_type}.json'))
        print('Graph saved to:', os.path.join(output_dir, f'graph_{event_type}_{user_type}.json'))
        if draw_graphs:
            draw_graph(
                graph,
                output_filename=os.path.join(output_dir, f'graph_{event_type}_{user_type}.html')
            )
            print('Graph saved to:', os.path.join(output_dir, f'graph_{event_type}_{user_type}.html'))
        name2graph[f'{event_type}_{user_type}'] = graph
    
    return name2graph


def create_graphs_from_path(
    input_path: str, output_dir: str, only_verbs_labels=True
) -> dict[str, nx.Graph]:
    '''
    Loads a CSV of parsed sentences dataframe and creates graphs for each movement and event types.
    '''
    df = pd.read_csv(
        input_path,
        converters={
            'parsed_labels': ast.literal_eval,
            'parsed_sentence': ast.literal_eval
        }
    )
    name2graph = split_df_create_graphs(
        df,
        output_dir,
        only_verbs_labels=only_verbs_labels
    )
    return name2graph