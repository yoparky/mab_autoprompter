from typing import Dict
import re
import string

def normalize_answer(s):

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

def exact_match(row_gt: Dict, llm_answer: str):
    gt = normalize_answer(str(row_gt["answer"]))
    llm_answer = normalize_answer(llm_answer)

    return 1.0 if gt == llm_answer else 0.0