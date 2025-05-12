from typing import Dict
import re
import string
from collections import Counter

def normalize_answer(s):
    print("\nnormalizing string...\n", s)

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_match(row_gt: Dict, llm_answer: str):
    print(type(row_gt["answer"]))
    print(type(llm_answer))
    gt = normalize_answer(str(row_gt["answer"]))
    llm_answer = normalize_answer(llm_answer)

    ZERO_METRIC = 0.0

    if llm_answer in ['yes', 'no', 'noanswer'] and llm_answer != gt:
        return ZERO_METRIC
    if gt in ['yes', 'no', 'noanswer'] and llm_answer != gt:
        return ZERO_METRIC

    prediction_tokens = llm_answer.split()
    ground_truth_tokens = gt.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    # return f1, precision, recall
    return round(f1, 3)
