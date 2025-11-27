import spacy
from tqdm import tqdm

NLP = spacy.load("en_core_web_sm")


def simple_coref_resolution(text):
    spans = []
    doc = NLP(text)
    for sent in doc.sents:
        for tok in sent:
            if tok.dep_ == "nsubj":
                span = doc[tok.left_edge.i : tok.right_edge.i+1]
                spans.append(span)
                #print(span.text)
    
    for span in spans:
        # They/It capitalized to make sure they are not replaced inside of a sentence
        if span.text == 'they' or span.text == 'it':
            text = text.replace(span.text, spans[0].text)
    
    return text


def resolve_in_batches(texts: list[str]) -> list[str]:
    docs = NLP.pipe(texts, batch_size=64, n_process=4)
    new_texts = []
    for doc in tqdm(docs, total=len(texts)):
        spans = [doc[tok.left_edge.i: tok.right_edge.i+1] for tok in doc if tok.dep_=="nsubj"]
        text = doc.text
        for span in spans:
            if span.text.lower() in ("they", "it", "she", "he"):
                text = text.replace(span.text, spans[0].text)
        new_texts.append(text)
    return new_texts


if __name__ == "__main__":
    text = (
        "Climate activists protest for climate justice. "
        "They demand that system change, not climate change. "
        "They stand for refugees, racial justice, and a sustainable planet."
    )

    assert simple_coref_resolution(text) == (
        'Climate activists protest for climate justice. '
        'Climate activists demand that system change, not climate change. '
        'Climate activists stand for refugees, racial justice, and a sustainable planet.'
    )

    text = (
        'The protesters emphasize that today matters for tomorrow. '
        'They advocate for keeping the Earth cool and protecting the planet. '
        'They warn that the planet is not okay and highlight the importance of urgent climate action.'
    )
    assert simple_coref_resolution(text) == (
        'The protesters emphasize that today matters for tomorrow. '
        'The protesters advocate for keeping the Earth cool and protecting the planet. '
        'The protesters warn that the planet is not okay and highlight the importance of urgent climate action.'
    )

    text = (
        'Sustainable United Neighborhoods (SUN) makes moves with partners like Pratt. '
        'They focus on social sustainability, green career pathways, and development. '
        'They advance meaningful GHG reductions through local Green Communities Initiatives.'
    )

    assert simple_coref_resolution(text) == (
        'Sustainable United Neighborhoods (SUN) makes moves with partners like Pratt. '
        'Sustainable United Neighborhoods (SUN) focus on social sustainability, green career pathways, and development. '
        'Sustainable United Neighborhoods (SUN) advance meaningful GHG reductions through local Green Communities Initiatives.'
    )

    # it
    text = (
        "The sign asks what is worth fighting for. "
        "It emphasizes fighting for the planet's future. "
        "It suggests that protecting the environment is more important than corporate profits and taxes."
    )
    print(simple_coref_resolution(text))
    assert simple_coref_resolution(text) == (
        "The sign asks what is worth fighting for. "
        "The sign emphasizes fighting for the planet's future. "
        "The sign suggests that protecting the environment is more important than corporate profits and taxes."
    )

    print(simple_coref_resolution('''The WWF states that sea levels are expected to rise by one meter by 2100 due to melting ice. The image shows a reindeer lying on the snow with melted ice around it, symbolizing the impact of climate change.'''))
