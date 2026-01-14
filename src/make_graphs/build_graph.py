from collections import defaultdict

import networkx as nx
import pandas as pd


NOT_NODE_LABELS = (
    'WHNP', 'TO', 'WHADVP', 'CC', 'RP', 'PRT', 'IN', 'RB', 'MD', 'DT'
)

NODES_TO_EXCLUDE = {
    'The image', 'The message', 'This image', 
    "This", "The images", 'The scene', "the country",
    'There', "The narrative", "others", "something", '', 'part',
    'less', 'co', "A man", "Despite", 'she', 'her', 'they', 'them', 'he', 
    'him', 'his', 'its', 'it'
}


def create_graph(df: pd.DataFrame, only_verbs_labels=True):
    '''
    Creates a directed multigraph from parsed sentences in a DataFrame.

    Nodes represent words (excluding common stop words and pronouns),
    and edges represent actions or relations, primarily verbs.
    Each node and edge stores metadata: sentence IDs and media IDs.

    Args:
        df (pd.DataFrame): Must contain columns 'sentence_id', 'media_id',
                           'parsed_sentence', and 'parsed_labels'.
        only_verbs_labels (bool): If True, keeps only edges whose labels are verbs.

    Returns:
        nx.MultiDiGraph: Graph with nodes and edges annotated with metadata.
    '''

    G = nx.MultiDiGraph()
    node2metadata = defaultdict(lambda: {'sentence_ids': set(), 'media_ids': set()})

    def extract_edges(parsed_sentence, parsed_labels):
        end = len(parsed_labels) - 1  # last token is "."
        i = 0
        while i < end:
            j = i + 1
            edge_label = ''
            edge_pos = None

            while j < end:
                pos = parsed_labels[j]
                tok = parsed_sentence[j]

                if pos.startswith('V') or pos == 'MD':
                    edge_label = tok
                    edge_pos = pos
                elif pos.isalpha() and pos not in NOT_NODE_LABELS:
                    break
                else:
                    if not edge_label and pos != 'DT':
                        edge_label = tok
                    elif pos in ('RB', 'MD') or pos.startswith('V'):
                        edge_label += ' ' + tok
                j += 1

            if j < len(parsed_sentence):
                yield (
                    parsed_sentence[i],
                    edge_label,
                    parsed_sentence[j],
                    edge_pos or parsed_labels[j - 1]
                )

            i = j

    for row in df.itertuples(index=False):
        sentence_id = row.sentence_id
        media_id = row.media_id
        parsed_labels = row.parsed_labels
        parsed_sentence = row.parsed_sentence

        for node1, edge_label, node2, edge_pos in extract_edges(
            parsed_sentence, parsed_labels
        ):
            if {node1, node2} & NODES_TO_EXCLUDE:
                continue
            if any('their' in n.lower() for n in (node1, node2)):
                continue
            if edge_pos == 'ADVP':
                continue

            edge_found = False
            if G.has_edge(node1, node2):
                for _, data in G[node1][node2].items():
                    if data.get('label') == edge_label:
                        data['weight'] += 1
                        data['sentence_ids'].add(sentence_id)
                        data['media_ids'].add(media_id)
                        edge_found = True
                        break

            if not edge_found:
                G.add_edge(
                    node1,
                    node2,
                    label=edge_label,
                    weight=1,
                    sentence_ids={sentence_id},
                    media_ids={media_id},
                    pos=edge_pos
                )

            for node in (node1, node2):
                node2metadata[node]['sentence_ids'].add(sentence_id)
                node2metadata[node]['media_ids'].add(media_id)

    for node, meta in node2metadata.items():
        if node in G:
            G.nodes[node]['sentence_ids'] = list(meta['sentence_ids'])
            G.nodes[node]['media_ids'] = list(meta['media_ids'])

    for _, _, _, data in G.edges(keys=True, data=True):
        data['sentence_ids'] = list(data['sentence_ids'])
        data['media_ids'] = list(data['media_ids'])

    if only_verbs_labels:
        to_remove = [
            (u, v, k)
            for u, v, k, d in G.edges(keys=True, data=True)
            if not str(d.get('pos', '')).startswith('V')
        ]
        G.remove_edges_from(to_remove)
        G.remove_nodes_from(list(nx.isolates(G)))

    return G






