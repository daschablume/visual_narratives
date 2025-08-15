import benepar
import pandas as pd
import spacy
from nltk import Tree

benepar.download('benepar_en3')

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("benepar", config={"model": "benepar_en3"})


def traverse_tree(tree: Tree, parsed: list) -> list:
    '''
    If the branch is a pure noun phrase, like "ice cover variations", it will be non chopped.
    Otherwise, it will be prased into smaller pieces.

    TODO1: parse CC into two different sentences.
    'The image shows street signs and a placard warning' =>
      'The image shows street signs'
      'The image shows a placard warning'
    
    Returns:
        a parsed list of tuples, like:
        [('NP', 'Presentation screen'), ('VBZ', 'displays'), ('NP', 'the logo')].
    '''
    for branch in tree:
        if branch.label() == "NP" and _contains_nested(branch):
            traverse_tree(branch, parsed)
        elif branch.label() in ("VP", "PP", "S", "SBAR", "ADJP"):
            traverse_tree(branch, parsed)
        else:
          joined_leaves = ' '.join(branch.leaves())
          parsed.append((branch.label(), joined_leaves))
    return parsed


def _contains_nested(np_branch):
    """
    Check if an NP contains nested structures:
        PP (prepositional clause) or SBAR (semantic dependent clause),
        S (dependent clause), or VP (verb phrase).
    """
    for leaf in np_branch:
        if leaf.label() in ("PP", "SBAR", "S", "VP"):
            return True
    return False
  

def clean_parsed(parsed: list[tuple]) -> tuple[list, list]:
    '''
    Clean the punctuation from the parsed list.
    '''
    parsed_sentence = []
    parsed_labels = []
    for label, piece in parsed:
        if not label.isalpha() or label == 'HYPH':
            continue
        parsed_labels.append(label)
        parsed_sentence.append(piece)
    return parsed_labels, parsed_sentence


def parse_sentence(sentence: str) -> tuple[list, list]:
    '''
    '''
    doc = nlp(sentence)
    sents = list(doc.sents)
    if not sents:
        return [], []

    tree = Tree.fromstring(sents[0]._.parse_string)
    parsed = traverse_tree(tree, [])
    parsed_labels, parsed_sentence = clean_parsed(parsed)
    return parsed_labels, parsed_sentence
