from enum import Enum
import os

import pandas as pd

from clustering.control_clusters import (
    rename_cluster, uncluster, merge_clusters, remove_from_cluster
)
from clustering.clusterize import replace_with_clusterized_labels
from create_graph import split_df_create_graphs, analyze_graphs

class Func(Enum):
    RENAME = 'rename'
    MERGE = 'merge'
    UNCLUSTER = 'uncluster'
    REMOVE = 'remove'


def reindex_graphs(
    func_name: str, folder: str, 
    curr_cluster_name: str=None,
    cluster_names: list[str]=None,
    phrases_to_remove: list[str]=None,
    new_cluster_name: str=None, 
    updated_df: pd.DataFrame = None, updated_df_path: str = None,
    verbs: bool=False,
    analyze=True
):
    '''
    A function to rename/merge/uncluster clusters, replace labels, and re-generate graphs and analysis.
    # TODO: move checking of df/path and verbs here from each correspondent function
    '''
    # TODO3: Enum for func
    # TODO4: I do not replace labels here, so the graphs will have old labels
    if func_name == Func.RENAME.value and not new_cluster_name and not curr_cluster_name:
        raise ValueError('Rename function can be called only with "new_cluster_name" variable')
    if func_name == Func.MERGE.value and not new_cluster_name and not cluster_names:
        raise ValueError('Merge function can be called only with "new_cluster_name" variable')
    if func_name == Func.UNCLUSTER.value and not curr_cluster_name:
        raise ValueError('Uncluster function can be called only with "curr_cluster_name" variable')
    if func_name == Func.REMOVE.value and not curr_cluster_name and not phrases_to_remove:
        raise ValueError('Remove function can be called only with "curr_cluster_name" and "phrases_to_remove" variables')
    
    if func_name == Func.RENAME.value:
        updated_df = rename_cluster(
            curr_cluster_name, folder, new_cluster_name, updated_df, updated_df_path, verbs
        )
    elif func_name == Func.MERGE.value:
        updated_df = merge_clusters(
            cluster_names, new_cluster_name, folder, updated_df, updated_df_path, verbs
        )
    elif func_name == Func.UNCLUSTER.value:
        updated_df = uncluster(
            curr_cluster_name, folder, updated_df, updated_df_path, verbs
        )
    elif func_name == Func.REMOVE.value:
        updated_df = remove_from_cluster(
            curr_cluster_name, phrases_to_remove, folder, updated_df, updated_df_path, verbs
        )
    else: 
        raise ValueError('Unknown func name')
    
    meta_path = os.path.join(
        folder, 'verbs_meta' if verbs else 'np_meta', 'clusters.csv'
    )
    # TODO: do I actually need to replace labels again? I think I do it anyway in each function
    #updated_df = replace_with_clusterized_labels(meta_path, updated_df)

    name2graph = split_df_create_graphs(updated_df, folder)
    if analyze:
        analyze_graphs(name2graph=name2graph)
    return name2graph


if __name__ == "__main__":
    name2graph = reindex_graphs(
        func_name=Func.UNCLUSTER.value,
        folder='../experiments5',
        curr_cluster_name='India and Russia',
        updated_df_path='../experiments5/updated_data.csv',
        verbs=False,
        analyze=True
    )