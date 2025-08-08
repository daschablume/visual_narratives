import math
import re

from allennlp.predictors.predictor import Predictor
import pandas as pd
from tqdm import tqdm

ROLES_PATT = re.compile(r'\[([^:]+):\s*([^\]]+)\]')

PREDICTOR = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz"
)


def _extract_roles_from_description(description) -> dict:
    """
    Extract semantic roles from AllenNLP description string.

    Returns:
        tag2role: a dictionary mapping tags to roles, like 
            {'Lights': 'illuminate', 'a basilica': 'ARG1'};
    """
    tags_roles = re.findall(ROLES_PATT, description)
    tag2role = {tag: role for tag, role in tags_roles}
    
    return tag2role


def predict_roles(df: pd.DataFrame, batch_size=64) -> pd.DataFrame:
    """
    The first variant of the function to predict SRL for a dataframe, preserving the ids.
    
    Batch_size is defined randomly for now.
    """
    total_batches = math.ceil(len(df) / batch_size)
    
    # iterate over the dataframe in batches
    # for each batch, predict roles 
    # for each batch, extract the df_id2image_id mapping
    # on each iteration, parse each batch: 
        # preserve the ids of images
        # create the ids of roles (for each extracted role)
        # extract the roles from the prediction
    
    roles = []
    
    # Create progress bar for batches
    batch_iterator = range(0, len(df), batch_size)
    progress_bar = tqdm(batch_iterator, 
                       desc="Processing SRL batches", 
                       unit="batch",
                       total=total_batches)
    
    for i in progress_bar:
        batch = df.iloc[i:i + batch_size]
        sentences = batch['sentence'].tolist()
        
        current_batch = (i // batch_size) + 1
        progress_bar.set_description(
            f"Processing batch {current_batch}/{total_batches} ({len(batch)} sentences)"
        )
        
        srl_results = PREDICTOR.predict_batch_json(
            [{"sentence": sentence} for sentence in sentences]
        )
        
        for j, result in enumerate(srl_results):
            image_id = batch.iloc[j]['id']
            for verb in result['verbs']:
                tag2role = _extract_roles_from_description(verb['description'])
                tags_lst = list(tag2role.keys())
                sentence_lst = list(tag2role.values())
                image_dicc = {
                    'image_id': image_id,
                    'sentence_lst': sentence_lst,
                    'ordered_roles': tags_lst
                }
                image_dicc.update(tag2role) 
                roles.append(image_dicc)
        
        progress_bar.set_postfix({
            'total_roles': len(roles),
            'sentences_processed': min(i + batch_size, len(df))
        })

    # TODO3: maybe change the model? But at first try with what we have here

    roles_df = pd.DataFrame(roles)
    roles_df['sentence_id'] = roles_df.index
    return roles_df
    
