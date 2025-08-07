import os

import pandas as pd
from relatio import SRL, extract_roles

from utils import read_tsv, Preprocessor, build_save_graph

OUTPUT_DIR = '/Users/macuser/Documents/visual-narratives/narr4'
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROCESSOR = Preprocessor()

df = read_tsv(path='prompts/prompt4.tsv')

df_sentences = PROCESSOR.split_into_sentences(df)

SRL_MODEL = SRL(
    path = "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz",
    batch_size = 10,
    cuda_device = -1
)

srl_res = SRL_MODEL(df_sentences['sentence'], progress_bar=True)
roles, _ = extract_roles(
    srl_res, 
    used_roles = ["ARG0","B-V","B-ARGM-NEG","B-ARGM-MOD","ARG1","ARG2"],
    only_triplets = True,
    progress_bar = True
)

# postprocessing of roles is skipped

build_save_graph(roles, path=os.path.join(OUTPUT_DIR, 'network_of_narratives.html'))


### TODO1: track the id of docs in the roles somehow

### TODO2: Activists in Madrid release green balloons symbolizing a call for climate action. =>
### Activists in Madrid => release => green. The rest is cropped! Not good.

# here => clean roles a bit

roles_df = pd.DataFrame(roles)

arg0_counts = roles_df['ARG0'].value_counts()
roles_df['arg0_frequency'] = roles_df['ARG0'].map(arg0_counts)

roles_df = roles_df.sort_values(['arg0_frequency', 'ARG0'], ascending=False)

arg0_to_exclude = ['The image', 'The infographic', 'The logo', 'The slide', 'The poster', 'The graphic']
clean_roles_df = roles_df[~roles_df['ARG0'].isin(arg0_to_exclude)]
# convert back to a list of dict
clean_roles = clean_roles_df.to_dict(orient='records')

build_save_graph(clean_roles, path=os.path.join(OUTPUT_DIR, 'clean_narratives.html'))
