import logging
from typing import Dict, List

from scipy.stats import beta as beta_dist

class MAB:
    def __init__(self, train_data, window_size=10):
        self.window_size = window_size
        self.test_data = {}
        for item in train_data:
            self.test_data[item["_id"]] = {
                'alpha': 1.0, # pseudo-success
                'beta': 1.0,  # pseudo-failure
                'results': [],
                'reasoning': "", # correct reasoning to reach gt
            }
        self.has_reasoning = set()

        self.params_data = {}

    
    def update_testcase_result(self, _id: str, reward: float):
        if _id not in self.test_data.keys():
            return
        reward = max(0.0, float(reward)) # prevent negative reward as it can mess up denominator
        self.test_data[_id]['results'].insert(0, reward)

        alpha_val = 1.0
        beta_val = 1.0
        if len(self.test_data[_id]['results']) < self.window_size:
            alpha_val += sum(self.test_data[_id]['results'])
            beta_val += sum(1.0 - r for r in self.test_data[_id]['results'])
        else: 
            for i in range(self.window_size):
                alpha_val += self.test_data[_id]['results'][i]
                beta_val += (1.0 - self.test_data[_id]['results'][i])
        
        self.test_data[_id]['alpha'] = alpha_val
        self.test_data[_id]['beta'] = beta_val

    def _hardness_sort_all_tests(self) -> List[str]: 
        # deterministically calculates hard tests based on current alpha beta values {_id: alpha / (alpha + beta)}, Hard Qs first
        result = {}
        for _id, testcase in self.test_data.items():
            alpha_val = testcase['alpha']
            beta_val = testcase['beta']
            # avoid divide by 0
            denominator = alpha_val + beta_val if (alpha_val + beta_val) > 0 else 0.5
            result[_id] = alpha_val / denominator

        return sorted(result, key=result.get)

    def _sample_sort_all_tests(self) -> List[str]: 
        # samples all of the tests based on the current alpha beta value {_id: sample}. Hard Qs first
        result = {}
        for _id, testcase in self.test_data.items():
            alpha_val = testcase['alpha']
            beta_val = testcase['beta']
            try:
                sample = beta_dist.rvs(alpha_val, beta_val, size=1)[0]
                result[_id] = sample
            except Exception as e: logging.error("Error running scipy beta.rvs")

        return sorted(result, key=result.get)
    
    def sample_k_hardest(self, k: int) -> List[str]:
        sampled_result = self._sample_sort_all_tests()
        if len(sampled_result) > k: sampled_result = sampled_result[:k]
        return sampled_result
    
    def stochastic_k_hardest(self, k: int) -> List[str]:
        stochastic_result = self._hardness_sort_all_tests()
        if len(stochastic_result) > k: stochastic_result = stochastic_result[:k]
        return stochastic_result
    
    def has_correct_reasoning_to_gt(self, _id: str):
        return _id in self.has_reasoning

    def add_correct_reasoning_to_gt(self, _id: str, correct_reasoning: str):
        self.test_data[_id]['reasoning'] = correct_reasoning
        self.has_reasoning.add(_id)

    def generate_k_shots_sample(self, train_set, train_set_lookup, k):
        formatted_shots_string_list = []
        shot_ids_list = self.sample_k_hardest(k)
        for _id in shot_ids_list:
            if _id in train_set_lookup.keys() and _id in self.has_reasoning:
                index = train_set_lookup[_id]
                context = train_set[index]["context"]
                answer = train_set[index]["answer"]
                reasoning = self.test_data[_id]["reasoning"]

                example = f"context: {context}\nground truth: {answer}\ncorrect reasoning: {reasoning}"
                formatted_shots_string_list.append(example)
        
        return formatted_shots_string_list

    def generate_k_shots_average(self, train_set, train_set_lookup, k):
        formatted_shots_string_list = []
        shot_ids_list = self.stochastic_k_hardest(k)
        for _id in shot_ids_list:
            if _id in train_set_lookup.keys() and _id in self.has_reasoning:
                index = train_set_lookup[_id]
                context = train_set[index]["context"]
                answer = train_set[index]["answer"]
                reasoning = self.test_data[_id]["reasoning"]

                example = f"context: {context}\nground truth: {answer}\ncorrect reasoning: {reasoning}"
                formatted_shots_string_list.append(example)
        
        return formatted_shots_string_list
    
    # PARAMS SECTION FROM HERE

    def initialize_params_data(self, params):
        for k, v in params.items():
            self.params_data[k] = {
                'alpha': 1.0, # pseudo-success
                'beta': 1.0,  # pseudo-failure
                'results': [],
            }
    def update_params_result(self, params_obj, score: float):
        if not params_obj: return
        reward = max(0.0, float(score))
        for k, v in params_obj.items():
            self.params_data[k]['results'].insert(0, reward)
            alpha_val = 1.0
            beta_val = 1.0
            if len(self.params_data[k]['results']) < self.window_size:
                alpha_val += sum(self.params_data[k]['results'])
                beta_val += sum(1.0 - r for r in self.params_data[k]['results'])
            else:
                for i in range(self.window_size):
                    alpha_val += self.params_data[k]['results'][i]
                    beta_val += (1.0 - self.params_data[k]['results'][i])
            self.params_data[k]['alpha'] = alpha_val
            self.params_data[k]['beta'] = beta_val

    def _hardness_sort_all_params(self) -> List[str]: 
        # deterministically calculates hard tests based on current alpha beta values {_id: alpha / (alpha + beta)}, Hard Qs first
        result = {}
        for k, param in self.params_data.items():
            alpha_val = param['alpha']
            beta_val = param['beta']
            # avoid divide by 0
            denominator = alpha_val + beta_val if (alpha_val + beta_val) > 0 else 0.5
            result[k] = alpha_val / denominator

        return sorted(result, key=result.get)

    def _sample_sort_all_params(self) -> List[str]: 
        # samples all of the tests based on the current alpha beta value {_id: sample}. Hard Qs first
        result = {}
        for k, param in self.params_data.items():
            alpha_val = param['alpha']
            beta_val = param['beta']
            try:
                sample = beta_dist.rvs(alpha_val, beta_val, size=1)[0]
                result[k] = sample
            except Exception as e: logging.error("Error running scipy beta.rvs")
            
        return sorted(result, key=result.get)