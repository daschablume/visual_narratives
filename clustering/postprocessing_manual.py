import os

from tqdm import tqdm
from relatio import FileLogger, Preprocessor, SRL, extract_roles, build_graph, draw_graph



sorted_groups = sorted(groups, key=len, reverse=True)[:30]
sorted_word_groups = [[narratives_clean[i] for i in group] for group in sorted_groups]


SRL = SRL(
    path = "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz",
    batch_size = 10,
    cuda_device = -1
)

folder = ''

narratives_path = os.path.join(folder, 'swe_narratives_cleaned.csv')
df = pd.read_csv(narratives_path)

narratives = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    freq = row['frequency']
    narrative = row['narrative']
    srl_res = SRL([narrative])
    roles, _ = extract_roles(
        srl_res, 
        used_roles = ["ARG0","B-V","B-ARGM-NEG","B-ARGM-MOD","ARG1","ARG2"],
        only_triplets = True,
    )
    i = 0
    while i < freq:
        narratives.extend(roles)
        i += 1

# convert narraves to df with columns [ARG0, B-V, B-ARGM-NEG, B-ARGM-MOD, ARG1, ARG2]
df = pd.DataFrame(narratives)

G = build_graph(
    df, 
    top_n = 20, 
    prune_network = True
)

graph_path = os.path.join(folder, 'experiment.html')
print('Drawing the graph')
draw_graph(
    G,
    notebook = True,
    show_buttons = False,
    width="1600px",
    height="1000px",
    output_filename = graph_path
)



def build_narrative(row):
    ARG0 = row['ARG0']
    V = row["B-V"]
    NEG = row["B-ARGM-NEG"]
    if pd.isna(NEG) == False:
        NEG = "not"

    MOD = row["B-ARGM-MOD"]
    ARG1 = row["ARG1"]
    ARG2 = row["ARG2"]

    pretty_narrative = (ARG0, MOD, NEG, V, ARG1, ARG2)
    pretty_narrative = " ".join([str(t) for t in pretty_narrative if (t != "" and pd.isna(t) == False)])
    
    return pretty_narrative


cleaned_roles = pd.read_csv(os.path.join(folder, 'clean_roles.csv'))
cleaned_roles = cleaned_roles[cleaned_roles['ARG0'].notna() & cleaned_roles['ARG1'].notna()]
cleaned_roles['narrative'] = cleaned_roles.apply(build_narrative, axis=1)

# Count the frequency of each narrative
narrative_counts = cleaned_roles['narrative'].value_counts().reset_index()
narrative_counts.columns = ['narrative', 'frequency']

# Sort the original dataframe based on the frequency of narratives
# First, create a mapping dictionary of narrative to frequency
narrative_freq_map = dict(zip(narrative_counts['narrative'], narrative_counts['frequency']))

# Add a frequency column to the original dataframe
cleaned_roles['narrative_frequency'] = cleaned_roles['narrative'].map(narrative_freq_map)

# Sort the dataframe by narrative frequency in descending order
sorted_df = cleaned_roles.sort_values(by='narrative_frequency', ascending=False)

# swedish sorted_df -- where narrative contains sweden or swedish
swedish_sorted_df = sorted_df[
    sorted_df['narrative'].str.contains("sweden|swedish", case=False, na=False)]

# exclude duplicates but keep the first occurrence
swedish_sorted_df = swedish_sorted_df.drop_duplicates(subset=['narrative'])

# Save the sorted dataframe to a CSV file
swedish_sorted_df.to_csv(os.path.join(folder, 'clean_sorted_swe_roles.csv'), index=False)

# let's say after the manual cleaning I read the same df again, delete the narrative columns,
# and repeat each row according to its frequency
df = pd.read_csv(os.path.join(folder, 'clean_sorted_swe_roles.csv'))
# repeat each row according to its frequency
df = df.loc[df.index.repeat(df['narrative_frequency'])].reset_index(drop=True)

G = build_graph(
    df, 
    top_n = 20, 
    prune_network = True
)

graph_path = os.path.join(folder, 'experiment.html')
print('Drawing the graph')
draw_graph(
    G,
    notebook = True,
    show_buttons = False,
    width="1600px",
    height="1000px",
    output_filename = graph_path
)

df = df.drop_duplicates(subset=['narrative'])
df.to_csv(os.path.join(folder, 'clean_sorted_swe_roles.csv'), index=False)


# pick a subset with migration, asylum, and refugee 
subset = df[df['narrative'].str.contains("migration|asylum|refugee", case=False, na=False)]

G = build_graph(
    subset, 
    top_n = 20, 
    prune_network = True
)

graph_path = os.path.join(folder, 'migrant.html')
print('Drawing the graph')
draw_graph(
    G,
    notebook = True,
    show_buttons = False,
    width="1600px",
    height="1000px",
    output_filename = graph_path
)

subset.drop_duplicates(subset=['narrative']).to_csv(os.path.join(folder, 'migrant.csv'), index=False)
subset = pd.read_csv(os.path.join(folder, 'migrant.csv'))
subset = subset.loc[subset.index.repeat(subset['narrative_frequency'])].reset_index(drop=True)

# filter out entities, which are in subset
not_migrant = df[~df['narrative'].isin(subset['narrative'])]
not_migrant = not_migrant.drop_duplicates(subset=['narrative'])
not_migrant.to_csv(os.path.join(folder, 'not_migrant.csv'), index=False)

G = build_graph(
    not_migrant, 
    top_n = 20, 
    prune_network = True
)

graph_path = os.path.join(folder, 'not_migrant.html')
print('Drawing the graph')
draw_graph(
    G,
    notebook = True,
    show_buttons = False,
    width="1600px",
    height="1000px",
    output_filename = graph_path
)

not_migrant = pd.read_csv(os.path.join(folder, 'not_migrant.csv'))
not_migrant = not_migrant.loc[not_migrant.index.repeat(not_migrant['narrative_frequency'])].reset_index(drop=True)
