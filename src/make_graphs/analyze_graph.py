import networkx as nx


def get_strength(graph, topn=20) -> tuple[
    dict[str, int], dict[str, int], dict[str, int]
]:
    '''
    Returns a top n nodes by strength (weighted degree), in-strength and out-strength.
    Non-normalized values.
    '''
    def _sort_di_view(item):
        return dict(sorted(item, key=lambda x: x[1], reverse=True)[:topn])
    strength = _sort_di_view(graph.degree(weight='weight'))
    in_strength = _sort_di_view(graph.in_degree(weight='weight'))
    out_strength = _sort_di_view(graph.out_degree(weight='weight'))
    return strength, in_strength, out_strength


def get_betweenness(graph: nx.Graph, topn=20) -> dict[str, float]:
    '''
    Returns a top n nodes by betweenness centrality -- weighted measure.
    OBS! Slow.
    Normalized values.
    '''
    betweenness = nx.betweenness_centrality(graph, weight='weight')
    return dict(sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:topn])


def get_strength_and_betweenness(graph: nx.Graph, topn:int=20) -> tuple[dict, dict, dict, dict]:
    '''
    Calls get_strength and get_betweenness.
    Returns strength, in_strength, out_strength, betweenness.
    '''
    strength, in_strength, out_strength = get_strength(graph, topn=topn)
    betweenness = get_betweenness(graph, topn=topn)
    return strength, in_strength, out_strength, betweenness


def get_strength_betweenness_multiple(name2graph: dict[str, nx.Graph], topn:int=20):
    '''
    Call get_strength_and_betweenness for multiple graphs.
    Returns a name2metrics_dict dictionary.
    '''
    name2metrics_dict = {}
    for name, graph in name2graph.items():
        metrics = get_strength_and_betweenness(graph, topn=topn)
        name2metrics_dict[name] = {
            'strength': metrics[0],
            'in_strength': metrics[1],
            'out_strength': metrics[2],
            'betweenness': metrics[3]
        }
    return name2metrics_dict


def print_metrics_table_for_graph(metrics: tuple[dict], with_numbers: bool = False):
    '''
    Print a metrics table for a single graph.

    Args:
        metrics: Tuple of four dicts (strength, in_strength, out_strength, betweenness).
        with_numbers: If True, include metric values; otherwise print ranked nodes only.
    '''
    metrics = {
        'strength': metrics[0],
        'in_strength': metrics[1],
        'out_strength': metrics[2],
        'betweenness': metrics[3]
    }
    columns = list(metrics.keys())
    header = " ".join(f"{col:<30}" for col in columns)
    print(header)
    print("-" * 120)

    all_keys = [list(metrics[col].keys()) for col in columns]
    max_len = max(len(keys) for keys in all_keys)

    for i in range(max_len):
        row_items = []
        for col_idx, col in enumerate(columns):
            keys = all_keys[col_idx]
            if i < len(keys):
                key = keys[i]
                if with_numbers:
                    item = f"{key}: {metrics[col][key]}"
                    row_items.append(f"{item:<30}")
                else:
                    row_items.append(f"{key:<30}")
            else:
                row_items.append(f"{'':<30}")
        print(" ".join(row_items))

