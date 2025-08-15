import os

from clustering.orchestrator import clusterize_srl, update_roles_with_clusters
from create_graph import create_graph, draw_graph, save_graph_to_json
from preprocessor import Preprocessor
from srl import predict_roles
from utils import read_tsv


OUTPUT_DIR = 'experiments3_better_graphs'


def main(input_file, output_dir):
    output_dir = os.path.join(OUTPUT_DIR, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    PROCESSOR = Preprocessor()

    # TODO: make sure the input is a TSV or change the func to be able to read CSV also
    df = read_tsv(input_file)

    df_sentences = PROCESSOR.split_into_sentences(
        df, output_path=os.path.join(output_dir, 'sentences.csv'))
    # WARNING: CHANGE THIS LATER
    df_sentences = df_sentences.iloc[:300]
    print("Warning! Working with the first 300 sentences!")

    roles_df = predict_roles(df_sentences)

    roles_df.to_csv(os.path.join(output_dir, 'roles.csv'), index=False)
    print(f"Roles saved to {os.path.join(output_dir, 'roles.csv')}")

    clusterize_srl(output_dir)
    print(f"Roles clustered and saved to {output_dir}")
    updated_roles_df = update_roles_with_clusters(output_dir)
    print(f"Roles updated with clusters and saved to {output_dir}")

    graph = create_graph(updated_roles_df)
    draw_graph(graph, output_filename=os.path.join(output_dir, 'network_of_narratives.html'))
    save_graph_to_json(graph, path=os.path.join(output_dir, 'graph.json'))

    return graph


if __name__ == "__main__":
    # Example usage
    main(input_file='prompts/prompt1.tsv', output_dir="prompt1_srl")
    main(input_file='prompts/prompt4.tsv', output_dir="prompt4_srl")


