import benepar
import pandas as pd
import spacy
from nltk import Tree

benepar.download('benepar_en3')

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("benepar", config={"model": "benepar_en3"})


def traverse_tree(tree: Tree, parsed: list) -> list:
    '''
    Simple list traversal.
    If the branch is a pure noun phrase, like "ice cover variations", it will be non chopped.
    Otherwise, it will be prased into smaller pieces.
    
    Returns:
        a parsed list of tuples, like:
        [('NP', 'Presentation screen'), ('VBZ', 'displays'), ('NP', 'the logo')].
    '''
    # TODO
    # People hold signs advocating for climate action. 
    #                           => ['People', 'hold', 'signs advocating for clima..']
    # 'Young adults and teenagers' (should be split)
    for branch in tree:
        if branch.label() == "NP" and _contains_nested_pp(branch):
            traverse_tree(branch, parsed)
        elif branch.label() == "VP" or branch.label() == "PP":
            traverse_tree(branch, parsed)
        else:
            joined_leaves = ' '.join(branch.leaves())
            parsed.append((branch.label(), joined_leaves))
    return parsed


def _contains_nested_pp(np_branch):
    """Check if an NP contains nested PP structures"""
    for leaf in np_branch:
        if leaf.label() == "PP":
            return True
    return False


def parse_sentence(sentence: str) -> tuple[list, list]:
    '''
    '''
    doc = nlp(sentence)
    tree = Tree.fromstring(next(doc.sents)._.parse_string)
    parsed = traverse_tree(tree, [])
    parsed_sentence = []
    parsed_labels = []
    for piece, label in parsed:
        parsed_sentence.append(piece)
        parsed_labels.append(label)
    return parsed_sentence, parsed_labels
