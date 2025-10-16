import ast
from collections import defaultdict
import csv
from enum import Enum
import os

import pandas as pd
from relatio.embeddings import Embeddings
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import spacy
from tqdm import tqdm

# TODO: improve lemmatization; now I call spacy for each word, which is slow
# move it to "simple coreference resolution"
NLP = spacy.load("en_core_web_sm")

# TODO: get rid of Relatio
EMBEDDING_MODEL = Embeddings(
    embeddings_type="SentenceTransformer",
    embeddings_model='all-MiniLM-L6-v2',
)

class Status(Enum):
    CLUSTERED = "clustered"
    UNCLUSTERED = "unclustered"
    RENAMED = "renamed"
    MERGED = "merged"
    SHRANK = "shrank"  # means when some phrases were removed manually


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
            

def embed_and_cluster(
    phrases: list[str], pca_args: dict = {'n_components': 50, 'svd_solver': 'full'},
    threshold: int = 0.7
) -> list[list[int]]:
    vectors = EMBEDDING_MODEL.get_vectors(phrases, progress_bar=True)
    # reduce dimensionality here with PCA
    pca_model = PCA(**pca_args).fit(vectors)
    training_vectors = pca_model.transform(vectors)

    clust = AgglomerativeClustering(
        metric="cosine",
        linkage="complete",
        distance_threshold=1 - threshold,
        n_clusters=None
    )
    labels = clust.fit_predict(training_vectors)
    clusters = defaultdict(list)
    for i, lbl in enumerate(labels):
        clusters[lbl].append(i)

    clusters = list(clusters.values())
    clusters = sorted(clusters, key=lambda x: -len(x))

    return clusters


def create_clusters(input_path: str, save_folder_path: str, 
    pca_args: dict = {'n_components': 50, 'svd_solver': 'full'}
):
    os.makedirs(save_folder_path, exist_ok=True)

    df = pd.read_csv(input_path)
    df['count'] = df.groupby('word')['word'].transform('count')
    phrase2count = df.drop_duplicates('word').set_index('word')['count'].to_dict()
    phrases = list(set(df['word'].tolist()))
    clusters = embed_and_cluster(phrases, pca_args)
    id2phrase = {idx: phrase for idx, phrase in enumerate(phrases)}
    
    with open(os.path.join(save_folder_path, 'clusters.csv'), 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'phrases', 'size', 'status', 'date_changed'])
        for cluster in tqdm(clusters):
            phrases = [id2phrase[idx] for idx in cluster]
            counts = [phrase2count[ph] for ph in phrases]
            label = phrases[counts.index(max(counts))]
            size = sum(counts)
            writer.writerow([label, phrases, size, Status.CLUSTERED, pd.Timestamp.now()])


def create_clusters_batched(
    input_path, save_folder_path, batch_size=15000,
    pca_args: dict = {'n_components': 50, 'svd_solver': 'full'}
):
    os.makedirs(save_folder_path, exist_ok=True)

    pre_clusters_path = os.path.join(save_folder_path, 'pre_clusters.csv')
    clusters_path = os.path.join(save_folder_path, 'clusters.csv')

    df = pd.read_csv(input_path)
    df['count'] = df.groupby('word')['word'].transform('count')
    phrase2count = df.drop_duplicates('word').set_index('word')['count'].to_dict()
    phrases = list(set(df['word'].tolist()))
    if len(phrases) <= batch_size:
        return create_clusters(input_path, save_folder_path, pca_args)

    with open(pre_clusters_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'phrases', 'size', 'status', 'date_changed'])
        for i in range(0, len(phrases), batch_size):
            batch = phrases[i:i+batch_size]
            print(f'Embedding batch {i//batch_size + 1} of {(len(phrases)-1)//batch_size + 1}')
            batch_clusters = embed_and_cluster(batch, pca_args)
            id2phrase = {idx: phrase for idx, phrase in enumerate(batch)}

            for cl in tqdm(batch_clusters): 
                cl_phrases = [id2phrase[idx] for idx in cl] 
                counts = [phrase2count[ph] for ph in cl_phrases] 
                cl_label = cl_phrases[counts.index(max(counts))] 
                cl_size = sum(counts)
                
                writer.writerow([
                    cl_label, cl_phrases, cl_size,
                    Status.CLUSTERED, pd.Timestamp.now()
                ])

    df = pd.read_csv(pre_clusters_path, converters={'phrases': ast.literal_eval})
    label2phrases = {row.label: row.phrases for row in df.itertuples(index=False)}
    
    clusters = embed_and_cluster(df['label'].tolist(), pca_args) 
    with open(clusters_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'phrases', 'size', 'status', 'date_changed'])
        for cluster in tqdm(clusters):
            df_subset = df.iloc[cluster]
            label2size = df_subset.groupby('label')['size'].sum().to_dict()
            label = max(label2size, key=label2size.get)
            size = sum(label2size.values())
            clustered_phrases = [
                phrase 
                for lab in df_subset['label']
                for phrase in label2phrases[lab]
            ]
            
            writer.writerow([label, clustered_phrases, size, Status.CLUSTERED, pd.Timestamp.now()])
            

    
def replace_with_clusterized_labels(meta_path: str, sentences_df: pd.DataFrame) -> pd.DataFrame:
    meta_df = pd.read_csv(meta_path, converters={'phrases': ast.literal_eval})
    phrase2label = {
        phr.lower(): tup.label 
        for tup in meta_df.itertuples() 
        for phr in tup.phrases 
        if tup.status != Status.UNCLUSTERED
    }

    sentences_df["parsed_sentence"] = sentences_df["parsed_sentence"].apply(
        lambda lst: [phrase2label.get(w.lower(), w) for w in lst]
    )

    return sentences_df


