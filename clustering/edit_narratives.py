import os

import pandas as pd
from relatio import build_graph, draw_graph


folder = '/Users/macuser/Documents/UPPSALA/thesis/results/relatio/sputnik_filtered_2010-2015_4+.csv'

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


df = pd.read_csv(os.path.join(folder, 'clean_roles.csv'))
df = df[df['ARG0'].notna() & df['ARG1'].notna()]
df['narrative'] = df.apply(build_narrative, axis=1)

# Count the frequency of each narrative
narrative_counts = df['narrative'].value_counts().reset_index()
narrative_counts.columns = ['narrative', 'frequency']

# Sort the original dataframe based on the frequency of narratives
# First, create a mapping dictionary of narrative to frequency
narrative_freq_map = dict(zip(narrative_counts['narrative'], narrative_counts['frequency']))

# Add a frequency column to the original dataframe
df['narrative_frequency'] = df['narrative'].map(narrative_freq_map)

# Sort the dataframe by narrative frequency in descending order
sorted_df = df.sort_values(by='narrative_frequency', ascending=False)

# swedish sorted_df -- where narrative contains sweden or swedish
swedish_sorted_df = sorted_df[
    sorted_df['narrative'].str.contains("sweden|swedish", case=False, na=False)]

# exclude duplicates but keep the first occurrence
swedish_sorted_df = swedish_sorted_df.drop_duplicates(subset=['narrative'])

# Save the sorted dataframe to a CSV file
swedish_sorted_df.to_csv(os.path.join(folder, 'clean_sorted_swe_roles.csv'), index=False)





# let's say after the manual cleaning I read the same df again, delete the narrative columns,
# and repeat each row according to its frequency
folder = '/Users/macuser/Documents/UPPSALA/thesis/results/relatio/euractiv_filtered_2010-2015_4+.csv'
source = folder.split('/')[-1].split('_')[0]
years = folder.split('/')[-1].split('_')[-2].split('-')
min_year, max_year = map(int, years)
if max_year != 2025:
    max_year = max_year - 1
graph_name = f"{source}{min_year}-{max_year}_not_pruned.html"

df = pd.read_csv(os.path.join(folder, 'clean_sorted_swe_roles.csv'))
# repeat each row according to its frequency
df = df.loc[df.index.repeat(df['narrative_frequency'])].reset_index(drop=True)

G = build_graph(
    df, 
    top_n = 50, 
    prune_network = False
)

graph_path = os.path.join(folder, graph_name)
print('Drawing the graph')
draw_graph(
    G,
    notebook = True,
    show_buttons = False,
    width="1600px",
    height="1000px",
    output_filename = graph_path
)

nato_subset = df[df['narrative'].str.contains("nato", case=False, na=False)]

# filter out nato => df without nato
not_nato =  df[~df['narrative'].isin(nato_subset['narrative'])]
not_nato.drop_duplicates(subset=['narrative']).to_csv(os.path.join(folder, 'not_nato.csv'), index=False)


not_nato = pd.read_csv(os.path.join(folder, 'not_nato.csv'))
# repeat each row according to its frequency
not_nato = not_nato.loc[not_nato.index.repeat(not_nato['narrative_frequency'])].reset_index(drop=True)

G = build_graph(
    not_nato, 
    top_n = 20, 
    prune_network = False
)

graph_path = os.path.join(folder, 'not_nato.html')
print('Drawing the graph')
draw_graph(
    G,
    notebook = True,
    show_buttons = False,
    width="1600px",
    height="1000px",
    output_filename = graph_path
)

quran_subset = df[df['narrative'].str.contains("quran", case=False, na=False)]

G = build_graph(
    quran_subset, 
    top_n = 20, 
    prune_network = True
)

graph_path = os.path.join(folder, 'quran_pruned.html')
print('Drawing the graph')
draw_graph(
    G,
    notebook = True,
    show_buttons = False,
    width="1600px",
    height="1000px",
    output_filename = graph_path
)




nato = pd.read_csv(os.path.join(folder, 'nato.csv'))
nato = nato.loc[nato.index.repeat(nato['narrative_frequency'])].reset_index(drop=True)

G = build_graph(
    nato, 
    top_n = 20, 
    prune_network = True
)

graph_path = os.path.join(folder, 'nato.html')
print('Drawing the graph')
draw_graph(
    G,
    notebook = True,
    show_buttons = False,
    width="1600px",
    height="1000px",
    output_filename = graph_path
)
