import ast
import os
import csv

import pandas as pd

from clustering.orchestrator import (
    process_and_cluster_phrases,
    update_sentences_with_clusterized
)
from create_graph import split_df_create_graphs, draw_graph
from preprocessor import Preprocessor
from simple_coref_resolution import resolve_in_batches


OUTPUT_DIR = '../experiments5'
PCA_ARGS = {'n_components': 50, 'svd_solver': 'full'}


def synt_parse_df(input_file='../experiments5/narr_finaldata_p11.tsv', output_dir="../experiments5"):
    from synt_parsing import parse_sentence
    '''
    This should be a part of main function.
    However, now I can't install "benepar" package.
    So instead, I split sentences and parse them in google colab.
    Then I save .csv file. So here, in main, I just read this .csv
    '''    
    PROCESSOR = Preprocessor()

    # TODO: make sure the input is a TSV or change the func to be able to read CSV also
    df = pd.read_csv(input_file, sep='\t')

    # filter out when the model fails to provide a narrative
    df = df[~df['Labels'].str.contains("clear narrative", case=False)]
    df = df[~df['Labels'].str.contains("no narrative", case=False)]
    df = df[~df['Labels'].str.contains("I can't provide", case=False)]
    df = df[~df['Labels'].str.contains("I can't extract", case=False)]
    df = df[~df['Labels'].str.contains("I can't determine", case=False)]
    df = df[~df['Labels'].str.contains("image does not", case=False)]

    # resolve coreferences: the resolution is quite primitive but resolves 94.889% of problems 
    # for "they"
    texts = df['Labels'].tolist()
    new_texts = resolve_in_batches(texts)
    df['Labels'] = new_texts

    df_sentences = PROCESSOR.split_into_sentences(
        df, output_path=os.path.join(output_dir, 'sentences.csv'))
    df_sentences = df_sentences.drop(columns=['Labels'])
    # filter out two-symbol sentences
    df_sentences = df_sentences[df_sentences['sentence'].str.len() > 2]
    df_sentences.to_csv(os.path.join(output_dir, 'sentences.csv'), index=False)

    # TODO: add parsing logic?
    output_file = os.path.join(output_dir, 'parsed_sentences.csv')
    # write to csv in order not to overload the memory
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['sentence_id', 'parsed_labels', 'parsed_sentence'] + df_sentences.columns.tolist()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in df_sentences.iterrows():
            labels, sentence = parse_sentence(row['sentence'])
            row_data = row.to_dict()
            writer.writerow({
                'sentence_id': idx,
                'parsed_labels': labels,
                'parsed_sentence': sentence,
                **row_data,
            })
    print('Parsed sentences saved to:', output_file)
    

def main(
    input_file, output_dir, 
    pca_args=PCA_ARGS,
):
    sentences_df = pd.read_csv(
        input_file,
        converters={
            'parsed_labels': ast.literal_eval,
            'parsed_sentence': ast.literal_eval
        }
    )
    sentences_df['path'] = sentences_df['Dir'].astype(str) + "/" + sentences_df['ImageID'].astype(str)

    # TODO: fix calling NLP from spacy on each verb + I do cluster verbs now
    process_and_cluster_phrases(sentences_df, output_dir, pca_args)
    print(f"Roles clustered and saved to {output_dir}")
    updated_roles_df = update_sentences_with_clusterized(sentences_df, output_dir)
    print(f"Roles updated with clusters and saved to {output_dir}")

    # merging with data 
    data = pd.read_csv(f'{output_dir}/data.tsv', sep='\t')
    merged_df = updated_roles_df.merge(data, on="path", how="inner")

    merged_df.to_csv(os.path.join(output_dir, 'updated_data.csv'), index=False)

    name2graph = split_df_create_graphs(merged_df, output_dir)
    for name, graph in name2graph.items():
        if name.endswith('_c'):
            draw_graph(
                graph, notebook=False,
                output_filename=os.path.join(output_dir, f'{name}_graph.html')
            )
            print('Graph is drawn and saved to:', os.path.join(output_dir, f'{name}_graph.html'))

if __name__ == "__main__":
    # Example usage; 194 components explain 90% variance
    main(
        input_file=f'../experiments6/parsed_sentences.csv', 
        output_dir='../experiments6', pca_args={'n_components': 194, 'svd_solver': 'full'}
)

