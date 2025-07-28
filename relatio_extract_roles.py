import os

import pandas as pd
import pickle as pk 
from relatio import Preprocessor, SRL, extract_roles
from relatio.utils import load_roles

OUTPUT_DIR = '/Users/macuser/Documents/visual-narratives'

output_sentences = os.path.join(OUTPUT_DIR, 'sentences.csv')
output_roles = os.path.join(OUTPUT_DIR, 'postproc_roles.json')
output_entities = os.path.join(OUTPUT_DIR, 'entities.pkl')
    
# image_id2desc is a dict {image_id: description.}

df = pd.DataFrame(image_id2desc.items(), columns=['id', 'doc'])

p = Preprocessor(
    spacy_model = "en_core_web_sm",
    remove_punctuation = True,
    remove_digits = True,
    lowercase = True,
    lemmatize = True,
    remove_chars = ["\"",'-',"^",".","?","!",";","(",")",",",":","\'","+","&","|","/","{","}",
                    "~","_","`","[","]",">","<","=","*","%","$","@","#","’"],
    stop_words = [],
    n_process = -1,
    batch_size = 100
)


SRL_MODEL = SRL(
    path = "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz",
    batch_size = 10,
    cuda_device = -1
)

print('Splitting into sentences, output path=', output_sentences)
df = p.split_into_sentences(
    df, output_path=None, progress_bar=True
)
df.to_csv(output_sentences)

print('Extracting roles')  ## TODO: why only 1000? !!!!!!!!!!!!!!!!!!!!!!!!!!!
srl_res = SRL_MODEL(df['sentence'][0:1000], progress_bar=True)
# returns a list of roles like "'Obamas', 'B-V': 'hate', 'ARG1': 'laced speeches'"
roles, sentence_index = extract_roles(
    srl_res, 
    used_roles = ["ARG0","B-V","B-ARGM-NEG","B-ARGM-MOD","ARG1","ARG2"],
    only_triplets = True,
    progress_bar = True
)

print('Printing the first 20 roles from extract_roles')
for d in roles[0:20]:
    print(d)


print('Postprocessing roles, output path=', output_roles)
postproc_roles = p.process_roles(
    roles, 
    max_length = 50,
    progress_bar = True,
    output_path = output_roles
)
print('Printing the first 20 postproc_roles')
for d in postproc_roles[0:20]:
    print(d)

postproc_roles = load_roles(output_roles)
