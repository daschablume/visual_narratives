import pandas as pd

from .clusterize import Status


def uncluster(
    cluster_name: str, unclustered_df: pd.DataFrame, clusters_df: pd.DataFrame,
    updated_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:

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

    clusters_df.loc[clusters_df['label'] == cluster_name, 'status'] = Status.UNCLUSTERED
    clusters_df.loc[clusters_df['label'] == cluster_name, 'unclustered_date'] = pd.Timestamp.now()

    return updated_df, clusters_df


def remove_from_cluster(
    cluster_name: str, phrases_to_remove: list[str], 
    unclustered_df: pd.DataFrame, clusters_df: pd.DataFrame,
    updated_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:

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
    
    # Create new row for removed phrases and append to clusters_df
    new_row = clusters_df[clusters_df['label'] == cluster_name].copy()
    new_row['phrases'] = [new_clustered_phrases]
    new_row['size'] = len(unclustered_df[unclustered_df['word'].isin(new_clustered_phrases)])
    new_row['status'] = Status.RENAMED
    new_row['unclustered_date'] = pd.Timestamp.now()

    # change the status of original cluster to Deleted
    clusters_df.loc[clusters_df['label'] == cluster_name, 'status'] = Status.DELETED
    clusters_df.loc[clusters_df['label'] == cluster_name, 'unclustered_date'] = pd.Timestamp.now()

    clusters_df = pd.concat([clusters_df, new_row], ignore_index=True)

    return updated_df, clusters_df


def rename_cluster(
    curr_cluster_name: str, new_cluster_name: str, 
    unclustered_df: pd.DataFrame, clusters_df: pd.DataFrame,
    updated_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:

    clustered_phrases = clusters_df[clusters_df['label'] == curr_cluster_name]['phrases'].values[0]
    to_update_df = unclustered_df[unclustered_df['word'].isin(clustered_phrases)]

    to_update_df = to_update_df.copy()
    to_update_df.loc[:, 'new_word'] = new_cluster_name

    updates_dict = to_update_df.groupby('sentence_id', group_keys=False).apply(
        lambda x: list(zip(x['position_idx'], x['new_word']))
    ).to_dict()

    def apply_updates(row):
        if row['sentence_id'] in updates_dict:
            parsed = row['parsed_sentence'].copy()
            for pos_idx, new_word in updates_dict[row['sentence_id']]:
                parsed[pos_idx] = new_word
            return parsed
        return row['parsed_sentence']

    updated_df['parsed_sentence'] = updated_df.apply(apply_updates, axis=1)

    # mark current cluster as DELETED
    clusters_df.loc[clusters_df['label'] == curr_cluster_name, 'status'] = Status.DELETED
    clusters_df.loc[clusters_df['label'] == curr_cluster_name, 'unclustered_date'] = pd.Timestamp.now()

    # add new cluster entry
    new_row = {
        'phrases': clustered_phrases,
        'label': new_cluster_name,
        'size': len(clustered_phrases),
        'status': Status.RENAMED,
    }
    clusters_df = pd.concat([clusters_df, pd.DataFrame([new_row])], ignore_index=True)

    return updated_df, clusters_df


def merge_clusters(
    cluster_names: list[str], new_cluster_name: str, 
    unclustered_df: pd.DataFrame, clusters_df: pd.DataFrame,
    updated_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:

    clustered_phrases = []
    for v in clusters_df[clusters_df['label'].isin(cluster_names)]['phrases'].values:
        clustered_phrases.extend(v)
    to_update_df = unclustered_df[unclustered_df['word'].isin(clustered_phrases)].copy()

    to_update_df.loc[:, 'new_word'] = new_cluster_name

    updates_dict = to_update_df.groupby('sentence_id', group_keys=False).apply(
        lambda x: list(zip(x['position_idx'], x['new_word'])),
        include_groups=False
    ).to_dict()

    def apply_updates(row):
        if row['sentence_id'] in updates_dict:
            parsed = row['parsed_sentence'].copy()
            for pos_idx, new_word in updates_dict[row['sentence_id']]:
                parsed[pos_idx] = new_word
            return parsed
        return row['parsed_sentence']

    updated_df['parsed_sentence'] = updated_df.apply(apply_updates, axis=1)

    # mark all old clusters as merged
    clusters_df.loc[clusters_df['label'].isin(cluster_names), 'status'] = Status.DELETED
    clusters_df.loc[clusters_df['label'].isin(cluster_names), 'unclustered_date'] = pd.Timestamp.now()

    # add new merged cluster
    new_row = {
        'label': new_cluster_name,
        'phrases': clustered_phrases,
        'size': len(to_update_df),
        'status': Status.MERGED,
        'unclustered_date': pd.NaT
    }
    clusters_df = pd.concat([clusters_df, pd.DataFrame([new_row])], ignore_index=True)            

    return updated_df, clusters_df
