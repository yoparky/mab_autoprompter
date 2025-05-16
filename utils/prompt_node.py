import uuid
import json
import copy
from llm_calls.regex_extractor import extract_demarcated_string

class PromptNode:
    def __init__(self, prompt, parent):
        self.prompt = prompt
        self.parent_id = None
        self.children_ids= []
        self.id = str(uuid.uuid4())
        self.has_expanded = False
        self.integrated_parameters = {}
        if parent:
            self.parent_id = parent.id
            self.integrated_parameters = copy.deepcopy(parent.integrated_parameters)
        
        self.validation_score = 0.0
        self.validation_rewards = []
        self.validation_answers = []
        self.validation_ids = []
        self.validation_input_tokens = 0
        self.validation_output_tokens = 0
        self.validation_literal_prompts = []
        self.validation_mapping = {} # id: {answer: , reward: , literal_prompt: }
        
        self.test_score = 0.0
        self.test_rewards = []
        self.test_answers = []
        self.test_ids = []
        self.test_input_tokens = 0
        self.test_output_tokens = 0
        self.test_literal_prompts = []
        self.test_mapping = {}

        self.train_score = 0.0
        self.train_rewards = []
        self.train_answers = []
        self.train_ids = []
        self.train_input_tokens = 0
        self.train_output_tokens = 0
        self.train_literal_prompts = []
        self.train_mapping = {}
        self.train_hard_analysis = {}


    # SETTER TUPLES: answers, ids, input_tokens, output_tokens, literal_prompt
    def set_validation_data(self, validation_raw_answers, start_demarcator, end_demarcator):
        self.validation_answers = [extract_demarcated_string(item, start_demarcator, end_demarcator) for item in validation_raw_answers[0]]
        self.validation_ids = validation_raw_answers[1]
        self.validation_input_tokens = validation_raw_answers[2][0] if validation_raw_answers[2][0] else 0
        self.validation_output_tokens = validation_raw_answers[3][0] if validation_raw_answers[3][0] else 0
        self.validation_literal_prompts = validation_raw_answers[4]

    def set_validation_mapping(self):
        for i, _id in enumerate(self.validation_ids):
            testcase = {}
            testcase["answer"] = self.validation_answers[i]
            testcase["reward"] = self.validation_rewards[i]
            testcase["literal_prompt"] = self.validation_literal_prompts[i]
            self.validation_mapping[_id] = testcase


    def set_test_data(self, test_raw_answers, start_demarcator, end_demarcator):
        self.test_answers = [extract_demarcated_string(item, start_demarcator, end_demarcator) for item in test_raw_answers[0]]
        self.test_ids = test_raw_answers[1]
        self.test_input_tokens = test_raw_answers[2][0] if test_raw_answers[2][0] else 0
        self.test_output_tokens = test_raw_answers[3][0] if test_raw_answers[3][0] else 0
        self.test_literal_prompts = test_raw_answers[4]
    
    def set_test_mapping(self):
        for i, _id in enumerate(self.test_ids):
            testcase = {}
            testcase["answer"] = self.test_answers[i]
            testcase["reward"] = self.test_rewards[i]
            testcase["literal_prompt"] = self.test_literal_prompts[i]
            self.test_mapping[_id] = testcase
    
    def set_train_data(self, train_raw_answers, start_demarcator, end_demarcator):
        self.train_answers = [extract_demarcated_string(item, start_demarcator, end_demarcator) for item in train_raw_answers[0]]
        self.train_ids = train_raw_answers[1]
        self.train_input_tokens = train_raw_answers[2][0] if train_raw_answers[2][0] else 0
        self.train_output_tokens = train_raw_answers[3][0] if train_raw_answers[3][0] else 0
        self.train_literal_prompts = train_raw_answers[4]

    def set_train_mapping(self):
        for i, _id in enumerate(self.train_ids):
            testcase = {}
            testcase["answer"] = self.train_answers[i]
            testcase["reward"] = self.train_rewards[i]
            testcase["literal_prompt"] = self.train_literal_prompts[i]
            self.train_mapping[_id] = testcase

    def update_parameters(self, raw_params: str):
        params_obj = {}
        if raw_params:
            params_obj = json.loads(raw_params)
        for k, v in params_obj.items():
            if k in self.integrated_parameters:
                self.integrated_parameters[k].append(v)
            else:
                self.integrated_parameters[k] = [v]

    def to_dict(self):
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "prompt": self.prompt,
            "children_ids": json.dumps(self.children_ids),
            "has_expanded": self.has_expanded,
            "integrated_parameters": json.dumps(self.integrated_parameters),

            "validation_score": self.validation_score,
            "validation_rewards": json.dumps(self.validation_rewards),
            "validation_answers": json.dumps(self.validation_answers),
            "validation_ids": json.dumps(self.validation_ids),
            "validation_input_tokens": self.validation_input_tokens,
            "validation_output_tokens": self.validation_output_tokens,
            "validation_literal_prompts": json.dumps(self.validation_literal_prompts),
            "validation_mapping": json.dumps(self.validation_mapping),

            "test_score": self.test_score,
            "test_rewards": json.dumps(self.test_rewards),
            "test_answers": json.dumps(self.test_answers),
            "test_ids": json.dumps(self.test_ids),
            "test_input_tokens": self.test_input_tokens,
            "test_output_tokens": self.test_output_tokens,
            "test_literal_prompts": json.dumps(self.test_literal_prompts),
            "test_mapping": json.dumps(self.test_mapping),

            "train_score": self.train_score,
            "train_rewards": json.dumps(self.train_rewards),
            "train_answers": json.dumps(self.train_answers),
            "train_ids": json.dumps(self.train_ids),
            "train_input_tokens": self.train_input_tokens,
            "train_output_tokens": self.train_output_tokens,
            "train_literal_prompts": json.dumps(self.train_literal_prompts),
            "train_mapping": json.dumps(self.train_mapping),
            "train_hard_analysis": json.dumps(self.train_hard_analysis),
        }

    def __lt__(self, other):
        if not isinstance(other, PromptNode): return NotImplemented
        return self.validation_score < other.validation_score
    
    def __eq__(self, other):
        if not isinstance(other, PromptNode): return NotImplemented
        return self.id == other.id