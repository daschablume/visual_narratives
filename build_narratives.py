import os
import csv

import pandas as pd

from clustering.orchestrator import (
    process_and_cluster_phrases,
    update_sentences_with_clusterized
)
from create_graph import create_graph, draw_graph, save_graph_to_json
from preprocessor import Preprocessor
from utils import read_tsv, load_lst_from_saved_txt


OUTPUT_DIR = '../experiments3_better_graphs'


def synt_parse_df(input_file='prompts/prompt4.tsv', output_dir="prompt4_synt_pars"):
    from synt_parsing import parse_sentence
    '''
    This should be a part of main function.
    However, now I can't install "benepar" package.
    So instead, I split sentences and parse them in google colab.
    Then I save .csv file. So here, in main, I just read this .csv
    '''    
    output_dir = os.path.join(OUTPUT_DIR, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    PROCESSOR = Preprocessor()

    # TODO: make sure the input is a TSV or change the func to be able to read CSV also
    df = read_tsv(input_file)

    df_sentences = PROCESSOR.split_into_sentences(
        df, output_path=os.path.join(output_dir, 'sentences.csv'))
    
    # filter out two-symbol sentences
    df_sentences = df_sentences[df_sentences['sentence'].str.len() > 2]
    # clean from "sorry" sentences -- when the model fails
    sentences_df = sentences_df[~sentences_df['sentence'].str.contains("sorry", case=False)]
    output_file = os.path.join(output_dir, 'parsed_sentences.csv')
    # write to csv in order not to overload the memory
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['image_id', 'sentence', 'parsed_labels', 'parsed_sentence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in df_sentences.iterrows():
            labels, sentence = parse_sentence(row['sentence'])
        
            writer.writerow({
                'image_id': row['id'],
                'sentence_id': idx,
                'sentence': row['sentence'],
                'parsed_labels': labels,
                'parsed_sentence': sentence
            })
    print('Parsed sentences saved to:', output_file)
    

def main(input_file, output_dir):
    output_dir = os.path.join(OUTPUT_DIR, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    sentences_df = pd.read_csv(input_file)
    #sentences_df = sentences_df.iloc[:300]
    #print("Warning! Working with the first 300 sentences!")
    sentences_df[['parsed_labels', 'parsed_sentence']] = load_lst_from_saved_txt(
        sentences_df, ['parsed_labels', 'parsed_sentence'])

    process_and_cluster_phrases(sentences_df, output_dir)
    print(f"Roles clustered and saved to {output_dir}")
    updated_roles_df = update_sentences_with_clusterized(sentences_df, output_dir)
    print(f"Roles updated with clusters and saved to {output_dir}")

    graph = create_graph(updated_roles_df)
    print('Graph created with', len(graph.nodes), 'nodes and', len(graph.edges), 'edges.')
    draw_graph(graph, output_filename=os.path.join(output_dir, 'network_of_narratives.html'))
    save_graph_to_json(graph, path=os.path.join(output_dir, 'graph.json'))
    print('Graph saved to:', os.path.join(output_dir, 'network_of_narratives.html'))

    return graph


if __name__ == "__main__":
    # Example usage
    main(input_file=f'{OUTPUT_DIR}/prompt1_synt_pars/parsed_sentences.csv', output_dir="prompt1_synt_pars_test")
    #main(input_file=f'{OUTPUT_DIR}/prompt4_synt_pars/parsed_sentences.csv', output_dir="prompt4_synt_pars")

