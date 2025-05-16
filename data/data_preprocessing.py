from typing import List, Dict
import random
import math
import pandas as pd
from datasets import load_dataset

def dataframe_to_list_of_dicts(df: pd.DataFrame) -> List[Dict]:
    """Converts a pandas DataFrame to a list of dictionaries."""
    if not isinstance(df, pd.DataFrame):
        return []
    return df.to_dict(orient='records')

def shuffle_test_set(dataset, random_seed: int = 13):
    split_random = random.Random(random_seed)
    shuffled_dataset = dataset[:]
    split_random.shuffle(shuffled_dataset)

    return shuffled_dataset

def split_data(dataset, test_ratio = 0.5, train_ratio = 0.3, val_ratio = 0.2):
    test_set, train_set, val_set = [], [], []

    # shuffling should be done outside if necessary
    curr_sum = test_ratio + train_ratio + val_ratio
    if (curr_sum < 1.0 or math.isclose(curr_sum, 1.0)) or not dataset:
        test_end_index = int(len(dataset) * test_ratio)
        train_end_index = int(len(dataset) * train_ratio) + test_end_index
        
        test_set = dataset[:test_end_index]
        train_set = dataset[test_end_index:train_end_index]
        val_set = dataset[train_end_index:]

    return test_set, train_set, val_set

def hf_dataset_to_list_of_dict(dataset_title, token, update_input={}): # TODO: add common answer if needed as arg
    hf_dataset = load_dataset(dataset_title, token=token)
    
    train = hf_dataset["train"].add_column("question", ["extract"] * len(hf_dataset["train"]))
    val = hf_dataset["eval"].add_column("question", ["extract"] * len(hf_dataset["eval"]))
    test = hf_dataset["test"].add_column("question", ["extract"] * len(hf_dataset["test"])) # we should use fill_non_existant_columns_uniformly to fill out the data

    if update_input:
        train = train.rename_columns(column_mapping=update_input)
        val = val.rename_columns(column_mapping=update_input)
        test = test.rename_columns(column_mapping=update_input)
        
    return list(train), list(val), list(test)
    
def fill_non_existant_columns_uniformly(list_of_dict, column_key_and_uniform_value):
    if not column_key_and_uniform_value:
        return
    for testcase in list_of_dict:
        for k, v in column_key_and_uniform_value.items():
            testcase[k] = v
    return

######################################################################
# Data specific

import json
def hpqa_filepath_to_list_of_testcases(file_path):
    """
    Converts SQuAD JSON data from a file to a pandas DataFrame.

    Args:
      file_path (str): The path to the SQuAD JSON file.

    Returns:
      pandas.DataFrame: A DataFrame where each row represents a question-answer entry.
                        Returns None if the file is not found or is not valid JSON.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            hpqa_list = json.load(f)
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return None
    return hpqa_list
    

def squad_json_to_dataframe_from_file(file_path):
    """
    Converts SQuAD JSON data from a file to a pandas DataFrame.

    Args:
      file_path (str): The path to the SQuAD JSON file.

    Returns:
      pandas.DataFrame: A DataFrame where each row represents a question-answer entry.
                        Returns None if the file is not found or is not valid JSON.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            squad_dict = json.load(f)
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return None

    rows_list = []

    for article in squad_dict['data']:
        title = article['title']
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question_id = qa['id']
                question_text = qa['question']
                is_impossible = qa['is_impossible']

                answer_texts = []
                answer_starts = []
                plausible_answer_texts = []
                plausible_answer_starts = []

                if not is_impossible:
                    for answer in qa.get('answers', []):
                        answer_texts.append(answer['text'])
                        answer_starts.append(answer['answer_start'])
                else:
                    if 'plausible_answers' in qa:
                        for plausible_answer in qa['plausible_answers']:
                            plausible_answer_texts.append(plausible_answer['text'])
                            plausible_answer_starts.append(plausible_answer['answer_start'])


                row = {
                    '_id': question_id,
                    'title': title,
                    'context': context,
                    'question': question_text,
                    'answer': ''.join(answer_texts).replace("[", "").replace("]",""),
                    'answers_start': str(answer_starts).replace("[", "").replace("]", ""),
                    'is_impossible': is_impossible,
                    'plausible_answers_text': ''.join(plausible_answer_texts).replace("[", "").replace("]", ""),
                    'plausible_answers_start': str(plausible_answer_starts).replace("[", "").replace("]", "")
                }
                rows_list.append(row)

    return pd.DataFrame(rows_list)

