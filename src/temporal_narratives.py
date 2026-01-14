from itertools import islice

import networkx as nx
import pandas as pd

from make_graphs.build_graph import create_graph
from make_graphs.analyze_graph import get_strength

COP = 'cop'
STRIKE = 'strike'
COUNTERMOVEMENT = 'c'
MOVEMENT = 'm'


def construct_day2graph(df: pd.DataFrame) -> dict:

    """    
    Creates a graph for each unique day in the DataFrame by filtering
    sentences belonging to that day and building a graph representation.
    """
    days = sorted(df['day'].unique())
    day2graph = {}
    for day in days:
        day_df = df[df['day'] == day]
        print(f"Creating graph for day {day} with {len(day_df)} sentences")
        graph = create_graph(day_df, only_verbs_labels=True)
        print('Graph created with', len(graph.nodes), 'nodes and', len(graph.edges), 'edges.')
        day2graph[day] = graph
    return day2graph


def construct_name2day2graph(merged_df: pd.DataFrame) -> dict:
    '''
    Construct nested dictionary mapping category names to day-to-graph mappings.
    
    Filters the merged DataFrame by event type (COP/STRIKE) and user type
    (MOVEMENT/COUNTERMOVEMENT), then creates graph mappings for each combination.
    '''
    cop_m = merged_df[(merged_df['event'] == COP) & (merged_df['usr_type'] == MOVEMENT)]
    cop_c = merged_df[(merged_df['event'] == COP) & (merged_df['usr_type'] == COUNTERMOVEMENT)]
    strike_m = merged_df[(merged_df['event'] == STRIKE) & (merged_df['usr_type'] == MOVEMENT)]
    strike_c = merged_df[(merged_df['event'] == STRIKE) & (merged_df['usr_type'] == COUNTERMOVEMENT)]

    name2day2graph = {}
    for df, name in [
        (cop_c, 'cop_c'),
        (cop_m, 'cop_m'),
        (strike_c, 'strike_c'),
        (strike_m, 'strike_m')
    ]:
        print(f"Processing {name} with {len(df)} sentences")
        day2graph = construct_day2graph(df)
        name2day2graph[name] = day2graph
    
    return name2day2graph


def get_top_narratives(graph: nx.Graph, max_weight: int = 2, topn: int = 10) -> list:
    '''
    Retrieves top narrative edges from a graph based on weight criteria.
    
    Returns edges with weight >= max_weight if specified, otherwise returns
    the top n edges by weight in descending order.
    
    Args:
        graph (nx.Graph): A NetworkX graph with weighted edges.
        max_weight (int, optional): Minimum weight threshold for edges. 
            If provided, returns all edges meeting this threshold. Defaults to 2.
        topn (int, optional): Number of top edges to return when max_weight 
            is None/falsy. Defaults to 10.
    
    Returns:
        list: List of tuples (u, v, data) representing graph edges sorted by 
            weight in descending order.
    
    Note:
        Recursively reduces max_weight by 1 if no edges meet the threshold,
        until max_weight reaches 1, at which point it falls back to computing
        narratives with maximum node strength.
    '''
    if max_weight:
        top_narr_dict = sorted([
            (u, v, data) 
            for u, v, data in graph.edges(data=True) 
            if data['weight'] >= max_weight
        ], key=lambda x: x[2]["weight"], reverse=True)
        if len(top_narr_dict) == 0:
            if max_weight == 1:
                print('No narratives found with weight >=', max_weight)
                print('Returning the narratives with max node strength')
                return _compute_narratives_with_max_node_strength(graph)
            print('No narratives found with weight >=', max_weight)
            print('Returning one level below, max_weight =', max_weight - 1)
            return get_top_narratives(graph, max_weight=max_weight - 1, topn=topn)
        print('Returning max_weight >=', max_weight, 'narratives:', len(top_narr_dict))
        return top_narr_dict
    print('Returning topn =', topn, 'narratives')
    return sorted(graph.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)[:topn]


def _compute_narratives_with_max_node_strength(graph: nx.Graph) -> list:
    '''
    Call this function inside get_top_narratives if all narratives have max weight=1.
    Computes narratives involving nodes with maximum strength.
    Returns a list of tuples (u, v, data) for edges where either node has maximum strength.
    '''
    strength, _, _ = get_strength(graph, topn=len(graph.nodes))
    max_strength = max(strength.values())
    strong_nodes = {n for n, s in strength.items() if s == max_strength}
    narratives = []
    for u, v, data in graph.edges(data=True):
        if data['weight'] == 1 and (u in strong_nodes or v in strong_nodes):
            narratives.append((u, v, data))
    return narratives


def get_clean_narratives(narratives: list[tuple], normalized: bool = False) -> dict:
    '''
    Extract and format narrative paths from graph edge data.
    
    Args:
        narratives: List of tuples containing (source_node, target_node, edge_data)
                   where edge_data is a dict with 'label', 'weight', and 
                   'normalized_weight' keys.
        normalized: If True, use normalized_weight instead of weight. 
                   Defaults to False.
    
    Returns:
        Sorted dictionary mapping narrative strings (format: "source - label -> target")
        to their corresponding weights.
    '''
    clean_narratives = {}
    for narr in narratives:
        u, v, data = narr
        label = data.get('label')
        if not label:
            print(f'Skipping nodes {u} -- {v} with no label')
            continue
        narrative = f'{u} - {label} -> {v}'
        if normalized:
            weight = data['normalized_weight']
        else:
            weight = data['weight']
        clean_narratives[narrative] = weight
    return dict(sorted(clean_narratives.items(), key=lambda x: x[1], reverse=True))


def get_clean_narratives_for_days(
    name2day2graph: dict[str, dict[str, nx.Graph]], 
    normalized: bool = False,
    max_weight: int = 2, cutoff: int = 20
) -> dict:
    """
    Extracts top narratives from multiple graphs organized by name and day.
    If normalized == True, uses normalized weights for narratives.
    
    Args:
        name2day2graph: Nested dictionary with structure {event_side: {day: graph}},
            i.e. {'cop_m': {1: graph1, 2: graph2, ...}, 'cop_c': {...}, ...}.
        max_weight: Maximum weight threshold for narrative extraction. 
                   Defaults to 2.
        cutoff: Maximum number of narratives to retain per day. Defaults to 20.
    
    Returns:
        Nested dictionary mapping event_sides to dictionaries of day-to-narratives,
        where narratives are dicts mapping narrative strings 
        "who - does what -> to whom" to weights.
    """
    name2day2narratives = {}
    for name, day2graph in name2day2graph.items():
        print('CALCULATING NARRATIVES FOR', name)
        day2narratives = {}
        for day, graph in day2graph.items():
            print(' Day:', day)
            narratives = get_top_narratives(graph, max_weight=max_weight)
            clean_narratives = get_clean_narratives(narratives, normalized=normalized)
            if len(clean_narratives) > cutoff:
                clean_narratives = dict(islice(clean_narratives.items(), cutoff))
            day2narratives[day] = clean_narratives
        name2day2narratives[name] = day2narratives
        print()
        print()
    return name2day2narratives


def get_top_narratives_normalized(
    graph: nx.Graph, min_weight: int = 2, topn: int = 10
) -> list:
    """    
    Retrieves edges (narratives) from the graph, filtering by minimum weight
    and computing normalized weights based on total edge count. If no edges
    meet the weight threshold, recursively reduces the threshold until edges
    are found or falls back to node strength computation.
    
    Args:
        graph: NetworkX graph containing narrative edges with 'weight' attribute.
        min_weight: Minimum edge weight threshold for filtering (default: 2).
        topn: Maximum number of top narratives to return (default: 10).
    
    Returns:
        List of tuples (u, v, data_dict) where data_dict includes both 'weight'
        and 'normalized_weight' keys. Sorted by weight in descending order.
    
    Note:
        When min_weight reaches 1 and no edges are found, falls back to
        _compute_narratives_with_max_node_strength_normalized().
    """
    total_narratives = graph.number_of_edges()

    def _norm(data):
        return data["weight"] / total_narratives if total_narratives else 0

    if min_weight:
        top_narr_dict = sorted([
            (u, v, {**data, "normalized_weight": _norm(data)})
            for u, v, data in graph.edges(data=True)
            if data["weight"] >= min_weight
        ], key=lambda x: x[2]["weight"], reverse=True)

        if len(top_narr_dict) == 0:
            if min_weight == 1:
                return _compute_narratives_with_max_node_strength_normalized(graph)
            return get_top_narratives_normalized(
                graph, max_weight=min_weight - 1, topn=topn
            )
        return top_narr_dict

    return sorted(
        [
            (u, v, {**data, "normalized_weight": _norm(data)})
            for u, v, data in graph.edges(data=True)
        ],
        key=lambda x: x[2]["weight"],
        reverse=True
    )[:topn]


def _compute_narratives_with_max_node_strength_normalized(graph: nx.Graph) -> list:
    """
    Computes narratives connected to nodes with maximum strength.
    
    Identifies edges with weight=1 that connect to at least one node having
    the maximum strength value in the graph. Adds normalized weights based
    on total edge count.
    """
    total_narratives = graph.number_of_edges()
    strength, _, _ = get_strength(graph, topn=len(graph.nodes))
    max_strength = max(strength.values())
    strong_nodes = {n for n, s in strength.items() if s == max_strength}

    narratives = []
    for u, v, data in graph.edges(data=True):
        if data["weight"] == 1 and (u in strong_nodes or v in strong_nodes):
            narratives.append(
                (u, v, {**data, "normalized_weight": data["weight"] / total_narratives})
            )
    return narratives

