import glob

import pandas as pd

from clusterize import prettify

path_pattern = '/Users/macuser/Documents/UPPSALA/thesis/results/relatio/*/clean_roles.csv'
file_paths = glob.glob(path_pattern)

for path in file_paths:
    folder_path = '/'.join(path.split('/')[:-1])
    source_parts = path.split('/')[-2].split('_')
    source = f'{source_parts[0]}, {source_parts[2]}'
    print(f'Processing {source}')
    df = pd.read_csv(path)   
    df = df.dropna(subset=['ARG0', 'ARG1', 'B-V'])

    narratives = []
    for _, row in df.iterrows():
        narratives.append(prettify(row))
    
    # group narratives by their frequency
    narratives = pd.Series(narratives).value_counts().reset_index()

    # write narratives to csv file
    narratives.columns = ['narrative', 'frequency']
    narratives.to_csv(f'{folder_path}/narratives.csv', index=False)
    
    # extract narratives about Sweden 
    swe_narratives = narratives[narratives['narrative'].str.contains('sweden|swedish', case=False, na=False)]

    # write narratives to csv file
    swe_narratives.to_csv(f'{folder_path}/swe_narratives.csv', index=False)