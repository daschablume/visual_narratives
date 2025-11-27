from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import spacy
NLP = spacy.load("en_core_web_sm")

from relatio import embeddings

POSTPROCESSING_SET = {
    'the image',
    'that',
    'this',
    'the magazine',
}

EDGE_LABELS = {'AUX', 'ADP', 'ADV', 'VERB', 'CCONJ'}


def dependency_graph(sentence):
    doc = NLP(sentence)

    # Merge noun chunks
    with doc.retokenize() as retokenizer:
        for chunk in doc.noun_chunks:
            retokenizer.merge(chunk)

    # Merge auxiliaries with their main verbs
    to_retokenize = []
    for token in doc:
        if token.dep_ == "aux" and token.head.pos_ == "VERB":
            start = min(token.i, token.head.i)
            end = max(token.i, token.head.i) + 1
            to_retokenize.append(doc[start:end])  # <-- span, not indices

    with doc.retokenize() as retokenizer:
        for span in to_retokenize:
            retokenizer.merge(span)

    dep_graph = [(token.head.text, token.text) for token in doc if token.head != token]

    # acomp

    pos_graph = [(token.head.pos_, token.pos_) for token in doc if token.head != token]

    return clean_dep_graph(dep_graph, pos_graph)


def clean_dep_graph(dep_graph, pos_graph):
    clean_indeces = []
    for idx, pair in enumerate(pos_graph):
        if 'PUNCT' not in pair:
            clean_indeces.append(idx)
    
    clean_dep = [dep_graph[i] for i in clean_indeces]
    clean_pos = [pos_graph[i] for i in clean_indeces]

    return clean_dep, clean_pos


def post_process_dep_graph(dep_graph, pos_graph):
    processed_indeces = []
    for idx, pair in enumerate(dep_graph):
        part1, part2 = pair
        if part1.lower() not in POSTPROCESSING_SET and part2.lower() not in POSTPROCESSING_SET:
            processed_indeces.append(idx)
    
    processed_dep = [dep_graph[i] for i in processed_indeces]
    processed_pos = [pos_graph[i] for i in processed_indeces]

    return processed_dep, processed_pos


def get_edge2node_prev(processed_dep, processed_pos):
    edge2nodes = defaultdict(list)
    edge_chains = []
    reserved_edge = None
    flag_next = False
    for pair, pos_pair in zip(processed_dep, processed_pos):
        word1, word2 = pair
        pos1, pos2 = pos_pair
        if reserved_edge:
            edge2nodes[reserved_edge].append(word2)
            reserved_edge = None
            flag_next = True
        elif pos1 in EDGE_LABELS and pos2 in EDGE_LABELS:
            edge2nodes[word1].append(None)
            edge2nodes[word2].append(None)
            edge_chains.append((word1, word2))
        elif pos1 in EDGE_LABELS:
            edge2nodes[word1].append(word2)
        elif pos2 in EDGE_LABELS:
            edge2nodes[word2].append(word1)
        elif pos1 in ('NOUN', 'PROPN') and pos2 == 'CCONJ': 
            # example: fear, and
            # 1) find a prev edge 
            # 2) add the curr node to the prev edge
            for edge, nodes in edge2nodes.items():
                for node in nodes:
                    if node == word1:
                        reserved_edge = edge
                        break
        elif flag_next:
            flag_next = False
            continue
        else:    
            print('A NEW EVENT ENCOUNTERED, PLEASE CHECK')
            print(word1, word2)
            print(pos1, pos2)

    return edge2nodes, edge_chains


def get_edge2node_prev_working(processed_dep, processed_pos):
    edge2nodes = defaultdict(list)
    edge_chains = []
    after_cconj = False
    for pair, pos_pair in zip(processed_dep, processed_pos):
        word1, word2 = pair
        pos1, pos2 = pos_pair
        if pos1 in EDGE_LABELS and pos2 in EDGE_LABELS:
            edge2nodes[word1].append(None)
            edge2nodes[word2].append(None)
            edge_chains.append((word1, word2))
        elif pos1 in EDGE_LABELS:
            edge2nodes[word1].append(word2)
        elif pos2 in EDGE_LABELS:
            edge2nodes[word2].append(word1)
            if pos2 == 'CCONJ':
                after_cconj = True
        elif after_cconj:
            after_cconj = False
            edge2nodes['and'].append(word2)
        else:    
            print('A NEW EVENT ENCOUNTERED, PLEASE CHECK')
            print(word1, word2)
            print(pos1, pos2)

    return edge2nodes, edge_chains


def get_edge2node(processed_dep, processed_pos):
    edge2nodes = defaultdict(list)
    edge_chains = []
    after_cconj = False
    for pair, pos_pair in zip(processed_dep, processed_pos):
        word1, word2 = pair
        pos1, pos2 = pos_pair
        if pos1 in EDGE_LABELS and pos2 in EDGE_LABELS:
            edge2nodes[word1].append(None)
            edge2nodes[word2].append(None)
            edge_chains.append((word1, word2))
        elif pos1 in EDGE_LABELS:
            edge2nodes[word1].append(word2)
        elif pos2 in EDGE_LABELS:
            edge2nodes[word2].append(word1)
            if pos2 == 'CCONJ':
                after_cconj = True
        elif after_cconj:
            after_cconj = False
            edge2nodes['and'].append(word2)
        else:    
            print('A NEW EVENT ENCOUNTERED, PLEASE CHECK')
            print(word1, word2)
            print(pos1, pos2)

    return edge2nodes, edge_chains


def postprocess_edge2node(edge2nodes, edge_chains):
    post_edge2nodes = dict()
    for edge1, edge2 in edge_chains:
        post_edge2nodes[edge1 + ' ' + edge2] = edge2nodes[edge1] + edge2nodes[edge2]

    cleaned_edge2nodes = dict()
    for edge, nodes in post_edge2nodes.items():
        cleaned_nodes = [node for node in nodes if node is not None]
        cleaned_edge2nodes[edge] = cleaned_nodes

    return cleaned_edge2nodes


def make_graph(edge2nodes):
    G = nx.MultiDiGraph()
    for edge, nodes in edge2nodes.items():
        if len(nodes) == 1:
            G.add_node(nodes[0])
        else:
            for i in range(len(nodes) - 1):
                node1, node2 = nodes[i], nodes[i+1]
                G.add_edge(node1, node2, label=edge)
    
    pos = nx.spring_layout(G)  # Get positions for all nodes
    nx.draw(G, pos, with_labels=True)
    
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.show()

    return G


def convert_to_graph_repr(edge2nodes):
    # TODO: skip edge/from/to
    graph_repr = []
    for edge, nodes in edge2nodes.items():
        if len(nodes) == 1:
            graph_repr.append({'node': nodes[0]})
        else:
            for i in range(len(nodes) - 1):
                node1, node2 = nodes[i], nodes[i+1]
                graph_repr.append({'edge': edge, 'from': node1, 'to': node2})
    return graph_repr


def parse(sentence):
    dep_graph, pos_graph = dependency_graph(sentence)
    processed_dep, processed_pos = post_process_dep_graph(dep_graph, pos_graph)
    edge2nodes, edge_chains = get_edge2node(processed_dep, processed_pos)
    if edge_chains:
        edge2nodes = postprocess_edge2node(edge2nodes, edge_chains)
    G = make_graph(edge2nodes)
    return G


    
if __name__ == "__main__":
    # Example
    s = "People are protesting on the stairs, holding signs about climate justice."
    print(dependency_graph(s))

# Problems:
# the graph looks nice but edges and nodes are the same thing here, it's unclear how to traverse/represent it
    
sentences = [
     'The climate movement profits from fear and manipulation',
     'Greta Thunberg and the climate movement are more about money than the environment.',
     'The magazine suggests Greta Thunberg is highly paid',
     'Climate change concerns may not be as urgent or proven as some claim.',
     'Alarmist predictions about climate disaster are often exaggerated.',
     'Youth participate in climate activism to raise awareness about climate justice.',
     'Youth demand urgent action to protect their future from climate change. ',
     'Youth see the need for intergenerational responsibility to address environmental issues.',
]