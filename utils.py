import csv
import json
import re
import time

import pandas as pd
import spacy
from spacy.cli import download as spacy_download
from tqdm import tqdm


def _clean_json_string(text):
    try:
        text = re.sub(r'^```json\s*', '', text)  # Remove ```json at start
        text = re.sub(r'^json\s*', '', text)     # Remove json at start
        text = re.sub(r'```$', '', text)         # Remove ``` at end
        text = text.replace("'", '"')            # Replace single quotes with double quotes
        return json.loads(text)
    except json.JSONDecodeError:
        return text
        

def read_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    # Check if any row in Labels column needs JSON cleaning
    if df['Labels'].str.contains('json|```').any():
        df['Labels'] = df['Labels'].apply(_clean_json_string)
        # TODO2: 65 out of 1000 have different format (strings), clean later
        string_mask = df['Labels'].apply(lambda x: isinstance(x, str))
        return df[~string_mask].reset_index(drop=True)
    return df

    

class Preprocessor:
    """
    A class to preprocess a given corpus
    (e.g., split it into sentences, clean)

    Args:
        spacy_model: One of the available spacy models for the English language (default: en_core_web_sm). For a complete list, see: https://spacy.io/models/en
        remove_punctuation: whether to remove string.punctuation
        remove_digits: whether to remove string.digits
        stop_words: list of stopwords to remove
        lowercase: whether to lower the case
        lemmatize: whether to lemmatize
        n_process: Number of processes to user in nlp.pipe() for parallel computing (default: -1). Set to -1 to use all cores on the machine.
        batch_size: Size of the batches for parallel computing (default: 1000 -- the SpaCy default).

    Note:
        self.nlp.add_pipe("sentencizer") => rule-based sentence segmentation without the dependency parse.
    """
    remove_chars = ["\"",'-',"^",".","?","!",";","(",")",",",":","\'","+","&","|","/","{","}",
                "~","_","`","[","]",">","<","=","*","%","$","@","#","’"]

    def __init__(
        self,
        spacy_model="en_core_web_sm",
        remove_punctuation: bool = True,
        remove_digits: bool = True,
        stop_words: list = [],
        lowercase: bool = True,
        lemmatize: bool = True,
        remove_chars: list = remove_chars,
        n_process: int = -1,
        batch_size: int = 1000,
    ):
        if not spacy.util.is_package(spacy_model):
            spacy_download(spacy_model)

        self.spacy_model = spacy_model
        self.nlp = spacy.load(spacy_model)
        self.nlp.add_pipe("sentencizer")
        self.n_process = n_process
        self.batch_size = batch_size
        self.remove_punctuation = remove_punctuation
        self.remove_digits = remove_digits
        self.stop_words = stop_words
        self.lowercase = lowercase
        self.lemmatize = lemmatize
        self.remove_chars = remove_chars

    def split_into_sentences(
        self,
        df: pd.DataFrame,
        output_path: str = None,
    ) -> pd.DataFrame:
        
        """
        Split a list of documents into sentences (using the SpaCy sentence splitter).

        Args:
            dataframe: a pandas dataframe with a column "ImageID" and a column "Labels"
            output_path: path to save the pandas DataFrame in a .csv format (default is None).
            NB: output_path is worth specifying for large datasets, since in this case, the result
            is written to a file row by row and then read back into a DataFrame.

        Returns:
            Pandas DataFrame with a column "image_id" and a column "sentence"

        """
        sentences = []
        image_ids = []

        length = len(df)
        # add 'id' column to extract just the id, without '.jpg' extension
        df['id'] = df['ImageID'].str.extract(r'(\d{1,})').astype(int)

        spacy_docs = self.nlp.pipe(
            df["Labels"],
            disable=["tagger", "ner", "parser", "lemmatizer"],
            batch_size=self.batch_size,
            n_process=self.n_process,
        )

        print("Splitting into sentences...")
        time.sleep(1)
        spacy_docs = tqdm(spacy_docs, total=length)

        if output_path is None:
            for i, doc in enumerate(spacy_docs):
                for sent in doc.sents:
                    sentences.append(str(sent))
                    image_ids = image_ids + [df["id"].iloc[i]]

            df = pd.DataFrame({"id": image_ids, "sentence": sentences}, index=None)

        else:
            with open(output_path, "w", newline="") as csvfile:
                fieldnames = ["id", "sentence"]
                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)
                for i, doc in enumerate(spacy_docs):
                    for sent in doc.sents:
                        doc_id = df["id"].iloc[i]
                        sentence = str(sent)
                        writer.writerow([doc_id, sentence])

            df = pd.read_csv(output_path)

        return df
