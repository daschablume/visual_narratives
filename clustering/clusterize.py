import ast
from collections import Counter
import csv
import os

import pandas as pd
from relatio.embeddings import Embeddings
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from tqdm import tqdm

from .graph_clustering import cluster_connected_components

# TODO: improve lemmatization; now I call spacy for each word, which is slow
NLP = spacy.load("en_core_web_sm")

# TODO: get rid of Relatio
EMBEDDING_MODEL = Embeddings(
    embeddings_type="SentenceTransformer",
    embeddings_model='all-MiniLM-L6-v2',
)


def prepare_df_to_clustering(df, verbs_csv_path=None, np_csv_path=None):
    '''
    Create CSV files for verbs and noun phrases from the DataFrame for clustering.
    Saves data item by item to avoid memory issues with large datasets.
    # TODO: the name is misleading, NP are prepared for clustering but 
    verbs are just lemmatized, I do not cluster them.
    '''
    if verbs_csv_path is None:
        verbs_csv_path = 'verbs_meta.csv'
        print('No verbs_csv_path provided, using default: verbs_meta.csv')
    if np_csv_path is None:
        np_csv_path = 'np_meta.csv'
        print('No np_csv_path provided, using default: np_meta.csv')

    with open(verbs_csv_path, 'w', newline='', encoding='utf-8') as verbs_file, \
         open(np_csv_path, 'w', newline='', encoding='utf-8') as np_file:
        
        verbs_writer = csv.DictWriter(verbs_file, fieldnames=['word', 'sentence_id', 'position_idx'])
        np_writer = csv.DictWriter(np_file, fieldnames=['word', 'sentence_id', 'position_idx'])
        
        verbs_writer.writeheader()
        np_writer.writeheader()
        
        for row in tqdm(df.itertuples(index=False), total=len(df)):
            labels, sentence = row.parsed_labels, row.parsed_sentence

            for idx, (label, part) in enumerate(zip(labels, sentence)):
                if not part:
                    continue
                
                if label.startswith('V'):
                    lemmatized = NLP(part)[0].lemma_
                    verbs_writer.writerow({
                        'word': lemmatized,
                        'sentence_id': row.sentence_id,
                        'position_idx': idx
                    })

                elif label.startswith('NN') or label == 'NP':
                    np_writer.writerow({
                        'word': part,
                        'sentence_id': row.sentence_id,
                        'position_idx': idx
                    })
    
    return verbs_csv_path, np_csv_path
            

def create_clusters_batched(input_path, save_folder_path, threshold=0.85, batch_size=15000):
    '''
    TODO: crappy clustering cause I almost manually add things to new files;
    TODO1: speficy tqdm for all loops, for similarity matrix, for the clustering function
    TODO2: please write ALL the names from the clusterized items in the final clusters.txt
    (now I write the names from pre-clusters only)
    '''
    os.makedirs(save_folder_path, exist_ok=True)

    # Paths for intermediate pre-cluster storage
    pre_clusters_path = os.path.join(save_folder_path, 'pre_clusters.csv')
    clusters_words_path = os.path.join(save_folder_path, 'clusters.txt')

    total_rows = sum(1 for _ in open(input_path)) - 1
    total_batches = (total_rows - 1) // batch_size + 1

    global_cluster_counter = 0

    # -----------------------------
    # Step 1: Process CSV in batches
    # -----------------------------
    for i, batch in enumerate(pd.read_csv(input_path, chunksize=batch_size)):
        print(f'Processing batch {i+1} of {total_batches}')
        batch_phrases = batch['word'].tolist()
        batch_vectors = EMBEDDING_MODEL.get_vectors(batch_phrases, progress_bar=True)
        assert len(batch_vectors) == len(batch_phrases)

        sim_matrix = cosine_similarity(batch_vectors)
        print('Grouping vectors')
        batch_groups = cluster_connected_components(sim_matrix, threshold)

        batch_index2phrase = dict(zip(batch.index, batch['word']))
        batch_index2sentence = dict(zip(batch.index, batch['sentence_id']))
        batch_index2position = dict(zip(batch.index, batch['position_idx']))

        remapped_groups = [
            [batch.index[idx] for idx in group]
            for group in batch_groups
        ]

        # Save pre-cluster info immediately (no memory storage)
        with open(pre_clusters_path, 'a') as f:
            for group in remapped_groups:
                phrases_in_group = [batch_index2phrase[idx] for idx in group]
                sentence_indices = [batch_index2sentence[idx] for idx in group]
                position_indices = [batch_index2position[idx] for idx in group]

                most_common_phrase = Counter(phrases_in_group).most_common(1)[0][0]

                f.write(f"{global_cluster_counter}\t{most_common_phrase}\t"
                        f"{len(phrases_in_group)}\t"
                        f"{sentence_indices}\t{position_indices}\n")
                global_cluster_counter += 1

    # -----------------------------
    # Step 2: Final clustering of pre-clusters (with batching)
    # -----------------------------
    print(f'Pre-clustering is finished with total {global_cluster_counter} clusters!')
    print('Creating final similarity matrix')
    
    
    if global_cluster_counter > batch_size:
        print(
            f'Pre-clusters exceed batch_size ({batch_size}), '
            'using batched approach for final clustering')
        
        print('Writing pre-cluster labels to memory')
        merged_clusters_counter = 0
        clusters_path = os.path.join(save_folder_path, 'pre_clusters_2.csv')
        # TODO: add tqdm
        # Process embeddings in batches to avoid memory issues
        for i, batch in enumerate(
            pd.read_csv(
                pre_clusters_path, header=None, 
                names=[
                    'pre_cluster_id', 'label', 'size',
                    'sentence_indices', 'position_indices'
                ],
                converters={
                    'sentence_indices': ast.literal_eval,
                    'position_indices': ast.literal_eval
                },
                chunksize=batch_size, sep='\t')):
            idx2pre_cluster_id = {idx: row.pre_cluster_id for idx, row in enumerate(batch.itertuples())}
            idx2pre_cluster_label = {idx: row.label for idx, row in enumerate(batch.itertuples())}
            print(f'Processing pre-cluster embeddings batch {i+1}')
            batch_labels = batch['label'].tolist()
            batch_vectors = EMBEDDING_MODEL.get_vectors(batch_labels, progress_bar=True)
            print('Creating similarity matrix for the batch')
            batch_sim_matrix = cosine_similarity(batch_vectors)
            batch_groups = cluster_connected_components(batch_sim_matrix, threshold)

            for pre_cluster_group in batch_groups:
                cluster_ids_to_merge = [idx2pre_cluster_id[idx] for idx in pre_cluster_group]
                merged_label = Counter(
                    [idx2pre_cluster_label[idx] for idx in pre_cluster_group]
                ).most_common(1)[0][0]
                merged_sentence_ids = batch['sentence_indices'].iloc[pre_cluster_group].sum()
                merged_position_ids = batch['position_indices'].iloc[pre_cluster_group].sum()

                with open(clusters_path, 'a') as f:
                    f.write(f"{merged_clusters_counter}\t{merged_label}\t"
                            f"{len(cluster_ids_to_merge)}\t"
                            f"{merged_sentence_ids}\t{merged_position_ids}\n")
                merged_clusters_counter += 1        
        print(
            f'Batched pre-clustering complete: {merged_clusters_counter} '
            f'merged clusters saved to {clusters_path}'
        )
        pre_clusters_final_path = clusters_path
        total_final_clusters = merged_clusters_counter
    else:
        pre_clusters_final_path = pre_clusters_path
        total_final_clusters = global_cluster_counter

    # -----------------------------
    # Step 2b: Global final clustering
    # -----------------------------
    print(f'Final clustering with {total_final_clusters} pre-clusters')
    all_labels = pd.read_csv(
        pre_clusters_final_path, header=None,
        names=['pre_cluster_id', 'label', 'size',
               'sentence_indices', 'position_indices'],
        converters={'sentence_indices': ast.literal_eval,
                    'position_indices': ast.literal_eval},
        sep='\t'
    )

    final_labels = all_labels['label'].tolist()
    final_vectors = EMBEDDING_MODEL.get_vectors(final_labels, progress_bar=True)
    print('Creating final similarity matrix')
    sim_matrix = cosine_similarity(final_vectors)
    print('Grouping final vectors')
    clusters_groups = cluster_connected_components(sim_matrix, threshold)

    # -----------------------------
    # Step 3: Create final clusters & save incrementally (no memory list)
    # -----------------------------
    clusters_words_path = os.path.join(save_folder_path, 'clusters.txt')
    cluster2label_path = os.path.join(save_folder_path, 'clusters.csv')
    
    with open(clusters_words_path, 'w') as f_words:
        first_write = True
        for cl_id, cl_group in enumerate(clusters_groups):
            cl_phrases, sentence_indices, position_indices = [], [], []

            for pre_cluster_id in cl_group:
                row = all_labels.loc[all_labels['pre_cluster_id'] == pre_cluster_id].iloc[0]
                cl_phrases.extend([row['label']] * row['size'])
                sentence_indices.extend(row['sentence_indices'])
                position_indices.extend(row['position_indices'])

            cl_label = Counter(cl_phrases).most_common(1)[0][0]

            # append row to CSV directly with pandas
            row_df = pd.DataFrame([{
                'cluster_id': cl_id,
                'label': cl_label,
                'cluster_size': len(cl_phrases),
                'sentence_indices': sentence_indices,
                'position_indices': position_indices
            }])
            mode = 'w' if first_write else 'a'
            header = first_write
            row_df.to_csv(cluster2label_path, index=False, sep=';', mode=mode, header=header)
            first_write = False

            f_words.write(f"CLUSTER{cl_id}\t{cl_label}\t(size: {len(cl_phrases)})\n")
            f_words.write(f"Words: {', '.join(cl_phrases)}\n")

    print(f'Clustering completed. Created {cl_id+1} clusters.')


def replace_with_clusterized_labels(clusters_df: pd.DataFrame, sentences_df: pd.DataFrame):
    sentence_id2idx = {sid: idx for idx, sid in enumerate(sentences_df['sentence_id'])}

    for row in clusters_df.itertuples(index=False):
        for position_id, sentence_id in zip(row.position_indices, row.sentence_indices):
            idx = sentence_id2idx[sentence_id]
            orig_sentence = sentences_df.at[idx, 'parsed_sentence']
            orig_sentence[position_id] = row.label
            sentences_df.at[idx, 'parsed_sentence'] = orig_sentence

    return sentences_df


def replace_with_lemmatized_verbs(lemmas_df: pd.DataFrame, sentences_df: pd.DataFrame):
    sentence_id2idx = {sid: idx for idx, sid in enumerate(sentences_df["sentence_id"])}

    for row in lemmas_df.itertuples(index=False):
        df_idx = sentence_id2idx[row.sentence_id]
        parsed = sentences_df.at[df_idx, "parsed_sentence"]
        parsed[row.position_idx] = row.word
        sentences_df.at[df_idx, "parsed_sentence"] = parsed

    return sentences_df

    