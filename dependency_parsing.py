import spacy

def dependency_graph(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)

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

    return [(token.head.text, token.text) for token in doc if token.head != token]

# Example
s = "People are protesting on the stairs, holding signs about climate justice."
print(dependency_graph(s))

