import ast
import os
import csv

import pandas as pd

from clustering.orchestrator import (
    process_and_cluster_phrases,
    update_sentences_with_clusterized
)
from create_graph import create_graph, draw_graph, save_graph_to_json, clean_graph
from preprocessor import Preprocessor


OUTPUT_DIR = '../experiments4'


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
    df = pd.read_csv(input_file, sep='\t')

    # filter out when the model fails to provide a narrative
    df = df[~df['Labels'].str.contains("clear narrative", case=False)]
    df = df[~df['Labels'].str.contains("no narrative", case=False)]
    df = df[~df['Labels'].str.contains("I can't provide", case=False)]
    df = df[~df['Labels'].str.contains("I can't extract", case=False)]
    df = df[~df['Labels'].str.contains("I can't determine", case=False)]
    df = df[~df['Labels'].str.contains("image does not", case=False)]

    df_sentences = PROCESSOR.split_into_sentences(
        df, output_path=os.path.join(output_dir, 'sentences.csv'))
    df_sentences = df_sentences.drop(columns=['Labels'])
    # filter out two-symbol sentences
    df_sentences = df_sentences[df_sentences['sentence'].str.len() > 2]

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
    

def main(input_file, output_dir):
    sentences_df = pd.read_csv(
        input_file,
        converters={
            'parsed_labels': ast.literal_eval,
            'parsed_sentence': ast.literal_eval
        }
    )
    sentences_df = sentences_df.drop(columns=['Labels'])
    sentences_df['path'] = sentences_df['Dir'].astype(str) + "/" + sentences_df['ImageID'].astype(str)

    # TODO: now clustering suffers from batching, not DRY code, not enough of 
    # messages about clustering progress; needs fixing, otherwise lead to suffering
    process_and_cluster_phrases(sentences_df, output_dir)
    print(f"Roles clustered and saved to {output_dir}")
    updated_roles_df = update_sentences_with_clusterized(sentences_df, output_dir)
    print(f"Roles updated with clusters and saved to {output_dir}")

    # merging with data 
    data = pd.read_csv(f'{output_dir}/data.tsv', sep='\t')
    merged_df = updated_roles_df.merge(data, on="path", how="inner")

    cop_c = merged_df[(merged_df['event'] == 'cop') & (merged_df['usr_type'] == 'c')]
    cop_m = merged_df[(merged_df['event'] == 'cop') & (merged_df['usr_type'] == 'm')]
    strike_c = merged_df[(merged_df['event'] == 'strike') & (merged_df['usr_type'] == 'c')]
    strike_m = merged_df[(merged_df['event'] == 'strike') & (merged_df['usr_type'] == 'm')]

    for df in [cop_c, cop_m, strike_c, strike_m]:
        event_type, user_type = df['event'].iloc[0], df['usr_type'].iloc[0]
        print(f"Event: {event_type}, User type: {user_type}, Number of sentences: {len(df)}")
        print('Creating graph')
        graph = create_graph(df)
        print('Graph created with', len(graph.nodes), 'nodes and', len(graph.edges), 'edges.')
        if len(df) > 10000:
            print('Huge dataset! Pruning the graph to its largest component.')
            cleaned_graph = clean_graph(graph)
        else:
            graph = clean_graph(graph, edge_weight=1)
        draw_graph(graph, output_filename=os.path.join(output_dir, f'graph_{event_type}_{user_type}.html'))
        save_graph_to_json(graph, path=os.path.join(output_dir, f'graph_{event_type}_{user_type}.json'))
        print('Graph saved to:', os.path.join(output_dir, f'graph_{event_type}_{user_type}.html'))


if __name__ == "__main__":
    # Example usage
    #main(input_file=f'{OUTPUT_DIR}/prompt1_synt_pars/parsed_sentences.csv', output_dir="prompt1_synt_pars_test")
    main(input_file=f'{OUTPUT_DIR}/parsed_sentences.csv', output_dir=OUTPUT_DIR)


