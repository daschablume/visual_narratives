import ast
import json
import re

import pandas as pd


def load_lst_from_saved_txt(df: pd.DataFrame, columns: list[str]):
    result = df[columns].copy()
    for col in columns:
        result[col] = result[col].apply(ast.literal_eval)
    return result


def read_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    # Check if any row in Labels column needs JSON cleaning
    if df['Labels'].str.contains('json|```').any():
        df['Labels'] = df['Labels'].apply(_clean_json_string)
        # TODO2: 65 out of 1000 have different format (strings), clean later
        string_mask = df['Labels'].apply(lambda x: isinstance(x, str))
        return df[~string_mask].reset_index(drop=True)
    return df


def _clean_json_string(text):
    try:
        text = re.sub(r'^```json\s*', '', text)  # Remove ```json at start
        text = re.sub(r'^json\s*', '', text)     # Remove json at start
        text = re.sub(r'```$', '', text)         # Remove ``` at end
        text = text.replace("'", '"')            # Replace single quotes with double quotes
        return json.loads(text)
    except json.JSONDecodeError:
        return text
