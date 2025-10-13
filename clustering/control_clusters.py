import ast
import os

import pandas as pd

from .clusterize import Status


def uncluster(
    cluster_name: str, folder: str,
    updated_df: pd.DataFrame = None, updated_df_path: str = None, verbs: bool=False
) -> pd.DataFrame:
    '''
    '''
    if updated_df_path and updated_df:
        raise ValueError("Can't have both updated_df_path and updated_df")
    if not updated_df_path and not updated_df:
        raise ValueError("Please provide either updated_df_path or updated_df")
    if updated_df_path:
        updated_df = pd.read_csv(
            updated_df_path,
            converters={
                'parsed_labels': ast.literal_eval,
                'parsed_sentence': ast.literal_eval
        })
    if verbs:
        cluster_path = os.path.join(folder, 'verbs_meta', 'clusters.csv')
        prepared_path = os.path.join(folder, 'verbs.csv')
    else:
        cluster_path = os.path.join(folder, 'np_meta', 'clusters.csv')
        prepared_path = os.path.join(folder, 'np.csv')
    
    unclustered_df = pd.read_csv(prepared_path)
    clusters_df = pd.read_csv(cluster_path, converters={'phrases': ast.literal_eval})

    clustered_phrases = clusters_df[clusters_df['label'] == cluster_name]['phrases'].values[0]
    to_update_df = unclustered_df[unclustered_df['word'].isin(clustered_phrases)]

    updates_dict = to_update_df.groupby('sentence_id', group_keys=False).apply(
        lambda x: list(zip(x['position_idx'], x['word'])),
        include_groups=False
    ).to_dict()

    def apply_updates(row):
        if row['sentence_id'] in updates_dict:
            parsed = row['parsed_sentence'].copy()
            for pos_idx, word in updates_dict[row['sentence_id']]:
                parsed[pos_idx] = word
            return parsed
        return row['parsed_sentence']

    updated_df['parsed_sentence'] = updated_df.apply(apply_updates, axis=1)

    # very primitive logging
    clusters_df.loc[clusters_df['label'] == cluster_name, 'status'] = Status.UNCLUSTERED
    clusters_df.loc[clusters_df['label'] == cluster_name, 'unclustered_date'] = pd.Timestamp.now()
    clusters_df.to_csv(cluster_path, index=False)

    if updated_df_path:
        updated_df.to_csv(updated_df_path, index=False)

    return updated_df


def remove_from_cluster(
    cluster_name: str, phrases_to_remove: list[str], folder: str,
    updated_df: pd.DataFrame = None, updated_df_path: str = None, verbs: bool=False
) -> pd.DataFrame:
    '''
    '''
    if updated_df_path and updated_df:
        raise ValueError("Can't have both updated_df_path and updated_df")
    if not updated_df_path and not updated_df:
        raise ValueError("Please provide either updated_df_path or updated_df")
    if updated_df_path:
        updated_df = pd.read_csv(
            updated_df_path,
            converters={
                'parsed_labels': ast.literal_eval,
                'parsed_sentence': ast.literal_eval
        })
    if verbs:
        cluster_path = os.path.join(folder, 'verbs_meta', 'clusters.csv')
        prepared_path = os.path.join(folder, 'verbs.csv')
    else:
        cluster_path = os.path.join(folder, 'np_meta', 'clusters.csv')
        prepared_path = os.path.join(folder, 'np.csv')
    
    unclustered_df = pd.read_csv(prepared_path)
    clusters_df = pd.read_csv(cluster_path, converters={'phrases': ast.literal_eval})

    clustered_phrases = clusters_df[clusters_df['label'] == cluster_name]['phrases'].values[0]
    to_update_df = unclustered_df[unclustered_df['word'].isin(phrases_to_remove)]

    updates_dict = to_update_df.groupby('sentence_id', group_keys=False).apply(
        lambda x: list(zip(x['position_idx'], x['word'])),
        include_groups=False
    ).to_dict()

    def apply_updates(row):
        if row['sentence_id'] in updates_dict:
            parsed = row['parsed_sentence'].copy()
            for pos_idx, word in updates_dict[row['sentence_id']]:
                parsed[pos_idx] = word
            return parsed
        return row['parsed_sentence']

    updated_df['parsed_sentence'] = updated_df.apply(apply_updates, axis=1)
    new_clustered_phrases = [ph for ph in clustered_phrases if ph not in phrases_to_remove]
    # TODO: double-check; works for now dunno if it's legal
    clusters_df.loc[clusters_df['label'] == cluster_name, 'phrases'][0] = new_clustered_phrases
    size = len(unclustered_df[unclustered_df['word'].isin(new_clustered_phrases)])
    clusters_df.loc[clusters_df['label'] == cluster_name, 'size'] = size

    # very primitive logging
    clusters_df.loc[clusters_df['label'] == cluster_name, 'status'] = Status.SHRANK
    clusters_df.loc[clusters_df['label'] == cluster_name, 'unclustered_date'] = pd.Timestamp.now()
    clusters_df.to_csv(cluster_path, index=False)

    if updated_df_path:
        updated_df.to_csv(updated_df_path, index=False)

    return updated_df


def rename_cluster(
    curr_cluster_name: str, new_cluster_name: str, folder: str, 
    updated_df: pd.DataFrame = None, updated_df_path: str = None, verbs: bool=False
) -> pd.DataFrame:
    '''
    TODO: restructure rename_cluster and unclusterize into one func, since it's almost the same code 
    '''
    if updated_df_path and updated_df:
        raise ValueError("Can't have both updated_df_path and updated_df")
    if not updated_df_path and not updated_df:
        raise ValueError("Please provide either updated_df_path or updated_df")
    if updated_df_path:
        updated_df = pd.read_csv(
            updated_df_path,
            converters={
                'parsed_labels': ast.literal_eval,
                'parsed_sentence': ast.literal_eval
        })
    if verbs:
        cluster_path = os.path.join(folder, 'verbs_meta', 'clusters.csv')
        prepared_path = os.path.join(folder, 'verbs.csv')
    else:
        cluster_path = os.path.join(folder, 'np_meta', 'clusters.csv')
        prepared_path = os.path.join(folder, 'np.csv')
    
    unclustered_df = pd.read_csv(prepared_path)
    clusters_df = pd.read_csv(cluster_path, converters={'phrases': ast.literal_eval})

    clustered_phrases = clusters_df[clusters_df['label'] == curr_cluster_name]['phrases'].values[0]
    to_update_df = unclustered_df[unclustered_df['word'].isin(clustered_phrases)]

    # TODO: pandas complain, please fix
    to_update_df['new_word'] = new_cluster_name

    updates_dict = to_update_df.groupby('sentence_id', group_keys=False).apply(
        lambda x: list(zip(x['position_idx'], x['new_word'])),
        include_groups=False
    ).to_dict()

    def apply_updates(row):
        if row['sentence_id'] in updates_dict:
            parsed = row['parsed_sentence']
            for pos_idx, new_word in updates_dict[row['sentence_id']]:
                parsed[pos_idx] = new_word
            return parsed
        return row['parsed_sentence']

    updated_df['parsed_sentence'] = updated_df.apply(apply_updates, axis=1)

    # very primitive logging
    clusters_df.loc[clusters_df['label'] == curr_cluster_name, 'status'] = Status.RENAMED
    clusters_df.loc[clusters_df['label'] == curr_cluster_name, 'unclustered_date'] = pd.Timestamp.now()
    clusters_df.to_csv(cluster_path, index=False)

    if updated_df_path:
        updated_df.to_csv(updated_df_path, index=False)

    return updated_df


def merge_clusters(
    cluster_names: list[str], new_cluster_name: str, folder: str, 
    updated_df: pd.DataFrame = None, updated_df_path: str = None, verbs: bool=False
) -> pd.DataFrame:
    '''
    TODO: restructure rename_cluster and unclusterize into one func, since it's almost the same code 
    '''
    if updated_df_path and updated_df:
        raise ValueError("Can't have both updated_df_path and updated_df")
    if not updated_df_path and not updated_df:
        raise ValueError("Please provide either updated_df_path or updated_df")
    if updated_df_path:
        updated_df = pd.read_csv(
            updated_df_path,
            converters={
                'parsed_labels': ast.literal_eval,
                'parsed_sentence': ast.literal_eval
        })
    if verbs:
        cluster_path = os.path.join(folder, 'verbs_meta', 'clusters.csv')
        prepared_path = os.path.join(folder, 'verbs.csv')
    else:
        cluster_path = os.path.join(folder, 'np_meta', 'clusters.csv')
        prepared_path = os.path.join(folder, 'np.csv')
    
    unclustered_df = pd.read_csv(prepared_path)
    clusters_df = pd.read_csv(cluster_path, converters={'phrases': ast.literal_eval})

    # TODO: make this more elegantly (list.extend)
    clustered_phrases = []
    for v in clusters_df[clusters_df['label'].isin(cluster_names)]['phrases'].values:
        clustered_phrases.extend(v)
    to_update_df = unclustered_df[unclustered_df['word'].isin(clustered_phrases)]

    # TODO: pandas complain, please fix
    to_update_df['new_word'] = new_cluster_name

    updates_dict = to_update_df.groupby('sentence_id', group_keys=False).apply(
        lambda x: list(zip(x['position_idx'], x['new_word'])),
        include_groups=False
    ).to_dict()

    def apply_updates(row):
        if row['sentence_id'] in updates_dict:
            parsed = row['parsed_sentence']
            for pos_idx, new_word in updates_dict[row['sentence_id']]:
                parsed[pos_idx] = new_word
            return parsed
        return row['parsed_sentence']

    updated_df['parsed_sentence'] = updated_df.apply(apply_updates, axis=1)

    # very primitive logging
    for cluster_name in cluster_names:
        clusters_df.loc[clusters_df['label'] == cluster_name, 'status'] = Status.MERGED
        clusters_df.loc[clusters_df['label'] == cluster_name, 'unclustered_date'] = pd.Timestamp.now()
    clusters_df.to_csv(cluster_path, index=False)

    if updated_df_path:
        updated_df.to_csv(updated_df_path, index=False)

    return updated_df