import pandas as pd
import graphviz
import os
import yaml

config_file_path = "./task_config.yaml"
with open(config_file_path, "r") as file:
    config = yaml.safe_load(file)
    
def visualize_prompt_tree(csv_path, output_filename="prompt_version_tree", output_format="png", view_after_render=True):
    """
    Generates a tree visualization from a CSV file containing prompt version data,
    coloring nodes based on the presence of test_score and train_score.

    - Green: Node has a test_score (regardless of train_score).
    - Blue: Node has a train_score but NO test_score.
    - Yellow (default): Node has neither test_score nor train_score.

    Args:
        csv_path (str): The path to the input CSV file.
        output_filename (str): The base name for the output file (without extension).
        output_format (str): The output format (e.g., 'png', 'pdf', 'svg').
        view_after_render (bool): Whether to automatically open the generated file.

    Returns:
        None: Generates output files (.gv source and the specified format).
    """
    try:
        df = pd.read_csv(csv_path, na_values=['', 'NaN', 'NA', 'nan', 'None', 'null'])
        print(f"Successfully read {len(df)} rows from CSV: {csv_path}")

        # node id
        if 'id' not in df.columns:
            return
        try:
            df['id'] = df['id'].astype(str)
        except Exception as e:
            return

        # parent id
        if 'parent_id' not in df.columns:
            return
        try:
            df['parent_id'] = df['parent_id'].apply(lambda x: str(x).strip() if pd.notna(x) else pd.NA)
            df['parent_id'] = df['parent_id'].replace('', pd.NA)
        except Exception as e:
            return

        # score columns
        score_column_map = {
            'test_score': 'internal_test_score',
            'validation_score': 'internal_validation_score',
            'train_score': 'internal_train_score'
        }

        for csv_col_name, internal_name in score_column_map.items():
            if csv_col_name in df.columns:
                df[internal_name] = pd.to_numeric(df[csv_col_name], errors='coerce')
            else:
                df[internal_name] = float('nan')

    except FileNotFoundError:
        return
    except pd.errors.EmptyDataError:
        return
    except Exception as e:
        return

    # Graph Initialization
    dot = graphviz.Digraph(comment='Prompt Version Tree', format=output_format)
    dot.attr(rankdir='TB')
    dot.attr('node', shape='box', style='rounded,filled', fontname='helvetica')
    dot.attr('edge', color='gray40')

    # Node and Edge Creation
    node_ids_in_csv = set(df['id'])
    nodes_added = 0
    edges_added = 0

    for index, row in df.iterrows():
        node_id_str = row['id']

        # Construct Node Label
        label_lines = [f"ID: {node_id_str}"]
        # Check for score existence using the internal names
        test_score_exists = pd.notna(row['internal_test_score']) and row['internal_test_score'] != 0.0
        train_score_exists = pd.notna(row['internal_train_score']) and row['internal_train_score'] != 0.0
        validation_score_exists = pd.notna(row['internal_validation_score'])

        if test_score_exists:
            label_lines.append(f"Test: {row['internal_test_score']:.2f}")
        if validation_score_exists:
            label_lines.append(f"Val: {row['internal_validation_score']:.2f}")
        if train_score_exists:
            label_lines.append(f"Train: {row['internal_train_score']:.2f}")

        node_label = "\n".join(label_lines)

        # node color based on score presence
        node_fill_color = 'lightyellow'  # Default
        if test_score_exists:
            node_fill_color = 'lightgreen'  # Green if test score exists (in final heal)
        elif train_score_exists:  # Blue if train and no test (was in heap, but was popped)
            node_fill_color = 'lightblue'

        dot.node(node_id_str, label=node_label, fillcolor=node_fill_color)
        nodes_added += 1

        # determine parent and add edge
        parent_id_from_csv = row['parent_id']

        if pd.notna(parent_id_from_csv):
            if parent_id_from_csv in node_ids_in_csv:
                dot.edge(parent_id_from_csv, node_id_str)
                edges_added += 1
            else:
                print(f"Warning: Parent ID '{parent_id_from_csv}' for node '{node_id_str}' not in id col")

    # render graph
    output_path_base = os.path.join(os.getcwd(), output_filename)
    print(f"Rendering graph to '{output_path_base}.{output_format}'...")
    try:
        rendered_path = dot.render(output_path_base, view=view_after_render, cleanup=True)
        print(f"Successfully rendered graph to: {rendered_path}")
    except graphviz.ExecutableNotFound:
        print("Graphviz executable not found. Visualization could not be generated.")

    except Exception as e:
        print(f"An error occurred during graph rendering: {e}")

if __name__ == "__main__":
    csv_file_path = config['output_file_path']
    output_base_name = config['output_tree_path']
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file does not exist: '{csv_file_path}'")
    else:
        visualize_prompt_tree(csv_file_path, output_base_name, output_format='png')