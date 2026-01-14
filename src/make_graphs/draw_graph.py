import matplotlib.colors as mcolors
import networkx as nx
from pyvis import network as net


GRAPH_REPULSION_PARAMS = {
    'node_distance': 120, 
    'central_gravity': 0.3, 
    'spring_length': 150, 
    'spring_strength': 0.08,
    'damping': 0.09
}

FONT_SETTINGS_PARAMS = {
    'face': 'arial',
    'vadjust': 0,  # Vertical adjustment
    'strokeWidth': 0,  # Remove text outline
    'strokeColor': '#ffffff'
}

MIN_NODE_SIZE = 15
MAX_NODE_SIZE = 40 
MIN_FONT_SIZE = 15 
MAX_FONT_SIZE = 30 

MIN_EDGE_WIDTH = 1
MAX_EDGE_WIDTH = 20  


ARROWS_PARAM = {
    'enabled': True,
    'scaleFactor': 0.5 
}

EDGE_COLOR_NORMAL = 'rgba(150, 150, 150, 0.4)'
EDGE_COLOR_HIGHLIGHT = 'rgba(100, 100, 100, 0.6)'
EDGE_COLOR_HOVER = 'rgba(100, 100, 100, 0.6)'

# Edge curve constants for multiple edges
CURVE_TYPE = 'curvedCW'
EDGE_CURVE_ROUNDNESS = {
    2: [-0.3, 0.3],
    3: [-0.4, 0, 0.4],
    4: [-0.5, -0.2, 0.2, 0.5]
}

MULTI_EDGE_MIN_ROUNDNESS = -0.6
MULTI_EDGE_MAX_ROUNDNESS = 0.6

# Default values for safety
DEFAULT_MAX_DEGREE = 1
DEFAULT_EDGE_WIDTH = MIN_EDGE_WIDTH


def draw_graph(
    networkx_graph,
    output_filename,
    notebook=True,
    width='1000px',
    height='1000px',
    show_buttons=False,
    only_physics_buttons=False,
    toggle_physics=True
):
    """
    This function accepts a networkx graph object,
    converts it to a pyvis network object preserving its node and edge attributes,
    and both returns and saves a dynamic network visualization.

    (For more info: https://pyvis.readthedocs.io/en/latest/documentation.html#pyvis.network.Network.add_node)

    Args:
        networkx_graph: The graph to convert and display
        output_filename: Where to save the converted network
        notebook: Display in Jupyter?
        width: width of the network
        height: height of the network
        show_buttons: Show buttons in saved version of network?
        only_physics_buttons: Show only buttons controlling physics of network?
        toggle_physics: Enable/disable physics simulation
    """
    if not networkx_graph.nodes():
        raise ValueError("Graph has no nodes")

    pyvis_graph = net.Network(notebook=notebook, directed=True)
    pyvis_graph.repulsion(**GRAPH_REPULSION_PARAMS)
    pyvis_graph.width = width
    pyvis_graph.height = height

    # Calculate weighted degree centrality and set node sizes
    degrees = dict(networkx_graph.degree(weight='weight'))
    max_degree = max(degrees.values()) if degrees else DEFAULT_MAX_DEGREE
    for node in networkx_graph.nodes():
        normalized_size = MIN_NODE_SIZE + (degrees[node] / max_degree) * (MAX_NODE_SIZE - MIN_NODE_SIZE)
        networkx_graph.nodes[node]['size'] = normalized_size
        
        # Add font size scaling based on node size with better proportions
        font_size = MIN_FONT_SIZE + (normalized_size - MIN_NODE_SIZE) / (MAX_NODE_SIZE - MIN_NODE_SIZE) * (MAX_FONT_SIZE - MIN_FONT_SIZE)
        
        # Enhanced font settings to prevent overlap
        networkx_graph.nodes[node]['font'] = {
            'size': int(font_size),
            **FONT_SETTINGS_PARAMS
        }        
        networkx_graph.nodes[node]['label'] = str(node)

    # Calculate betweenness centrality for node coloring
    betweenness = nx.betweenness_centrality(networkx_graph, weight='weight')
    
    # Normalize betweenness values to [0,1] for color mapping
    if betweenness and networkx_graph.nodes(): 
        min_bet = min(betweenness.values())
        max_bet = max(betweenness.values())
        bet_range = max_bet - min_bet
        
        for node in networkx_graph.nodes():
            # Normalize to [0,1]
            if bet_range > 0:
                normalized_bet = (betweenness[node] - min_bet) / bet_range
            else:
                normalized_bet = 0
            
            # Map to red→blue color gradient (high betweenness = red, low = blue)
            color = mcolors.to_hex((normalized_bet, 0, 1 - normalized_bet))  # (R,G,B)
            networkx_graph.nodes[node]['color'] = color
    
    # Process edges only if they exist
    if networkx_graph.edges():
        # Calculate edge thickness based on weight
        edge_weights = [edge_attrs['weight'] for _, _, edge_attrs in networkx_graph.edges(data=True)]
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        weight_range = max_weight - min_weight

        edge_count = {}
        for source, target, edge_attrs in networkx_graph.edges(data=True):
            edge_key = (source, target)
            if edge_key not in edge_count:
                edge_count[edge_key] = 0
            edge_count[edge_key] += 1

        edge_counter = {}

        for source, target, edge_attrs in networkx_graph.edges(data=True):
            # Normalize weight to width range
            if weight_range > 0:
                normalized_width = MIN_EDGE_WIDTH + (
                    (edge_attrs['weight'] - min_weight) / weight_range) * (MAX_EDGE_WIDTH - MIN_EDGE_WIDTH)
            else:
                normalized_width = DEFAULT_EDGE_WIDTH
            
            edge_attrs['width'] = normalized_width
            
            # Add transparency to edges
            edge_attrs['color'] = {
                'color': EDGE_COLOR_NORMAL,
                'highlight': EDGE_COLOR_HIGHLIGHT,
                'hover': EDGE_COLOR_HOVER
            }
            
            edge_attrs['arrows'] = {
                'to': {**ARROWS_PARAM}
            }
            
            # Handle multiple edges from same source to same target (preserving direction)
            edge_key = (source, target)
            if edge_key not in edge_counter:
                edge_counter[edge_key] = 0
            
            current_edge_num = edge_counter[edge_key]
            total_edges = edge_count[edge_key]
            
            # If multiple edges exist from this source to this target, apply smooth curves
            if total_edges > 1:
                # Get predefined roundness values or calculate for many edges
                if total_edges in EDGE_CURVE_ROUNDNESS:
                    roundness_values = EDGE_CURVE_ROUNDNESS[total_edges]
                else:
                    # For more edges, create evenly spaced curves
                    roundness_range = MULTI_EDGE_MAX_ROUNDNESS - MULTI_EDGE_MIN_ROUNDNESS
                    roundness_values = [
                        MULTI_EDGE_MIN_ROUNDNESS + (roundness_range * i / (total_edges - 1)) 
                        for i in range(total_edges)
                    ]
                
                edge_attrs['smooth'] = {
                    'enabled': True,
                    'type': CURVE_TYPE,
                    'roundness': roundness_values[current_edge_num]
                }
            
            edge_counter[edge_key] += 1

    # Add nodes 
    for node, node_attrs in networkx_graph.nodes(data=True):
        pyvis_graph.add_node(node, **node_attrs)

    # Add edges 
    for source, target, edge_attrs in networkx_graph.edges(data=True):
        pyvis_graph.add_edge(source, target, **edge_attrs)

    # Configure display options
    if show_buttons:
        if only_physics_buttons:
            pyvis_graph.show_buttons(filter_=["physics"])
        else:
            pyvis_graph.show_buttons()

    pyvis_graph.toggle_physics(toggle_physics)

    return pyvis_graph.show(output_filename)
