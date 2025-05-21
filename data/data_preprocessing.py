from typing import List, Dict
import uuid
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

def hf_dataset_to_list_of_dict(dataset_title, split_name="train", config_name="", token="", trust_remote_code=False, rename_column_mapping={}, add_column_mapping={}, map_function=None, add_id=False):
    dataset = None
    if config_name:
        dataset = load_dataset(dataset_title, config_name, token=token, trust_remote_code=trust_remote_code)
    else:
        dataset = load_dataset(dataset_title, token=token, trust_remote_code=trust_remote_code)

    dataset_split = dataset[split_name]
    dataset_split = dataset_split.rename_columns(column_mapping=rename_column_mapping)
    for k, v in add_column_mapping.items():
        new_column = [v] * len(dataset_split)
        dataset_split = dataset_split.add_column(k, new_column)
    if add_id:
        id_col = []
        for i in range(len(dataset_split)):
            id_col.append(str(uuid.uuid4()))
        dataset_split = dataset_split.add_column('_id', id_col)
        
    if map_function:
        dataset_split = dataset_split.map(map_function)
    
    dataset_list = list(dataset_split)
    return dataset_list
######################################################################
# TwinDoc, HotpotQA (bdsaglam/hotpotqa-distractor), BBH, LogiQA, GSM8k (openai/gsm8k)

# TwinDoc
# dataset = hf_dataset_to_list_of_dict("TwinDoc/GIEI2", split_name="train", token="???", rename_column_mapping={'text' : 'context', 'label': 'answer', 'filename': '_id'}, add_column_mapping={"question": ""})

# ARC
# dataset = hf_dataset_to_list_of_dict("allenai/ai2_arc", split_name="train", config_name='ARC-Easy', token="???", rename_column_mapping={'choices' : 'context', 'answerKey': 'answer', 'id': '_id'})

# BBH
# dataset = hf_dataset_to_list_of_dict("maveriq/bigbenchhard", split_name="train", config_name='causal_judgement', token="???", rename_column_mapping={'input' : 'context', 'target': 'answer'}, add_id=True)

# PiQA - requires custom input [question, choice_0, choice_1, answer]
# dataset = hf_dataset_to_list_of_dict("ybisk/piqa", split_name="train", token="???", rename_column_mapping={'goal' : 'question', 'sol1': 'choice_0', 'sol2': 'choice_1', 'label': 'answer'}, add_id=True, trust_remote_code=True)

# LogiQA - requires custom input [context, question, options, answer]
# dataset = hf_dataset_to_list_of_dict("lucasmccabe/logiqa", split_name="train", token="???", rename_column_mapping={'query' : 'question', 'correct_option': 'answer'}, add_id=True)

# DROP
# dataset = hf_dataset_to_list_of_dict("ucinlp/drop", split_name="train", token="???", rename_column_mapping={'query_id' : '_id', 'passage': 'context'}, map_function=(lambda ex: {**ex, 'answer': ex['answers_spans']['spans']}))

# JSONSchemaBench - requires custom input [json_schema]
# dataset = hf_dataset_to_list_of_dict("epfl-dlab/JSONSchemaBench", split_name="train", token="???", rename_column_mapping={'unique_id' : '_id'})

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


def jsonl_filepath_to_list_of_testcases(file_path):
    """
    Converts JSONL data from a file to a list of dict.

    Args:
      file_path (str): The path to the SQuAD JSON file.

    Returns:
      list of test cases (List[Dict])
    """
    testcases = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                testcase = json.loads(line)
                testcase['_id'] = str(uuid.uuid4())
                testcase['context'] = ""
                testcases.append(testcase)
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return None
    return testcases

