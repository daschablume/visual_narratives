import ast
import os
import csv

import pandas as pd

from clustering.orchestrator import (
    process_and_cluster_phrases,
    update_sentences_with_clusterized
)
from make_graphs.utils import split_df_create_graphs, draw_graph
from preprocessor import Preprocessor
from coref_resolution import resolve_in_batches
from synt_parsing import parse_sentence


PCA_ARGS = {'n_components': 50, 'svd_solver': 'full'}


def preprocess_sentences(input_file: str, output_dir: str):
    '''
    Input_file: .tsv file with columns: 'Dir', 'ImageID', 'Labels'
    Output_dir: directory to save the processed sentences

    1. Read the input file 
    2. Filter out invalid descriptions (where the model failed to provide one)
    3. Resolve coreferences
    4. Parse each description into sentences
    5. Parse each sentence with a syntactic parser (here: benepar)
    '''
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    PROCESSOR = Preprocessor()

    df = pd.read_csv(input_file, sep='\t')
    # filter out when the model fails to provide a narrative
    df = df[~df['Labels'].str.contains("clear narrative", case=False)]
    df = df[~df['Labels'].str.contains("no narrative", case=False)]
    df = df[~df['Labels'].str.contains("I can't", case=False)]
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
    print('Sentences saved to:', os.path.join(output_dir, 'sentences.csv'))
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
    return output_file
    

def build_narratives_from_parsed_df(
    input_file: str, output_dir: str, 
    pca_args: dict=PCA_ARGS,
):
    sentences_df = pd.read_csv(
        input_file,
        converters={
            'parsed_labels': ast.literal_eval,
            'parsed_sentence': ast.literal_eval
        }
    )
    sentences_df['path'] = sentences_df['Dir'].astype(str) + "/" + sentences_df['ImageID'].astype(str)

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
                graph,
                output_filename=os.path.join(output_dir, f'{name}_graph.html')
            )
            print('Graph is drawn and saved to:', os.path.join(output_dir, f'{name}_graph.html'))


def build_narratives(input_file: str, output_dir: str, pca_args: dict=PCA_ARGS):
    '''
    Input_file: .csv file with step2 descriptions
    Output_dir: directory to save the narratives
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    parsed_sentences_path = preprocess_sentences(input_file, output_dir)

    build_narratives_from_parsed_df(
        input_file=parsed_sentences_path,
        output_dir=output_dir,
        pca_args=pca_args
    )
        

if __name__ == "__main__":
    INPUT_FILE = 'data/data.tsv'  # input .tsv file with columns: 'Dir', 'ImageID', 'Labels'
    OUTPUT_DIR = 'output_narratives'  # directory to save the narratives

    build_narratives(INPUT_FILE, OUTPUT_DIR)
