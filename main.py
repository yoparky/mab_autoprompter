import yaml
import asyncio
import sys
import heapq

from llm_provider import create_llm_instance
from data_preprocessing import dataframe_to_list_of_dicts, squad_json_to_dataframe_from_file, split_data
from semaphore_call import batch_unified_call
from prompt_node import PromptNode
from metrics.exact_match import exact_match
from metrics.f1_match import f1_match
from regex_extractor import extract_demarcated_string
from mab import MAB

async def main():
    config_file_path = "./task_config.yaml"
    squad_file_path = "./squad-train-v2.0.json"

    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)

    # must make sure datasets have a unique "_id" column
    dataset = dataframe_to_list_of_dicts(squad_json_to_dataframe_from_file(squad_file_path))
    dataset = dataset[:40]
    test_set, train_set, val_set = split_data(dataset)
    
    # llm setup
    student_llm = create_llm_instance(config["llm_provider"], config["llm_provider_model"], float(config["llm_provider_temperature"]))
    generator_llm = create_llm_instance(config["llm_provider"], config["llm_generator_model"], float(config["llm_generator_temperature"]))

    # concurrency
    max_concurrent_calls = 2
    semaphore = asyncio.Semaphore(max_concurrent_calls)

    # heap
    node_heap = []
    heapq.heapify(node_heap)

    # Test case logic start. Can use same logic for test set
    val_set_lookup = {}
    for i, item in enumerate(val_set):
        val_set_lookup[item["_id"]] = i

    # MAB
    mab = MAB(train_set)

    # list of prompts
    next_prompt = config["test_prompt"]

    for k in range(5):
        # prompt
        prompt_node = PromptNode(next_prompt, None)
        # @ VALIDATION
        input_dict = {"context": "context", "question": "question"}
        answer = await batch_unified_call(student_llm, semaphore, val_set, prompt_node.prompt, input_dict)
        prompt_node.set_validation_data(answer, '---ANSWER_START---', '---ANSWER_END---')

        for item in answer[0]:
            print(item)

        rewards = []
        for i, item in enumerate(prompt_node.validation_answers):
            reward = f1_match(val_set[val_set_lookup[prompt_node.validation_ids[i]]], prompt_node.validation_answers[i])
            rewards.append(reward)

        val_score = round(sum(rewards)/len(rewards) if len(rewards) > 0 else 0.0, 4)
        prompt_node.validation_rewards = rewards
        prompt_node.validation_score = val_score

        prompt_node.set_validation_mapping()

        print(prompt_node.validation_rewards)
        print(prompt_node.validation_score)

        ### This should be parameterized. Ensures heap only tracks top k elements
        # This should also be only done with validation sets.
        if len(node_heap) < 5: # k = 4
            heapq.heappush(node_heap, prompt_node)
        else:
            heapq.heappushpop(node_heap, prompt_node)
        # Test case logic end, can use same logic for test set
        # For train set, there is a departure in that we CAN use this, but starting with the first call, we need a unified
        # data structure that stores the id: correct reasoning trajectory, so we do not have to make the same calls over and over again
        # @ TRAIN
        train_set_lookup = {}
        for i, item in enumerate(train_set):
            train_set_lookup[item["_id"]] = i

        
        input_dict = {"context": "context", "question": "question"}
        answer = await batch_unified_call(student_llm, semaphore, train_set, prompt_node.prompt, input_dict)
        prompt_node.set_train_data(answer, '---ANSWER_START---', '---ANSWER_END---')
        rewards = []

        for item in answer[0]:
            print(item)

        for i, item in enumerate(prompt_node.train_answers):
            reward = f1_match(train_set[train_set_lookup[prompt_node.train_ids[i]]], prompt_node.train_answers[i])
            mab.update_testcase_result(prompt_node.train_ids[i], reward)
            rewards.append(reward)

        train_score = round(sum(rewards)/len(rewards) if len(rewards) > 0 else 0.0, 4)
        prompt_node.train_rewards = rewards
        prompt_node.train_score = train_score

        prompt_node.set_train_mapping()

        print(prompt_node.train_rewards)
        print(prompt_node.train_score)

        # CORRECT REASONING INFERENCE PART (dynamically add not-processed parts)
        # sample hard qs
        hardest_cases = mab.sample_k_hardest(5) # list of str, [_id]
        list_of_testcases_to_analyze = []
        for item in train_set:
            if item['_id'] in hardest_cases and not mab.has_correct_reasoning_to_gt(item['_id']):
                list_of_testcases_to_analyze.append(item)
        
        if list_of_testcases_to_analyze:
            analysis_prompt = config["analyze_correct_reasoning_prompt"]
            input_dict = {"context": "context", "question": "question", "answer": "answer"}
            answer = await batch_unified_call(student_llm, semaphore, list_of_testcases_to_analyze, analysis_prompt, input_dict)

            for i, item in enumerate(answer[0]):
                reasoning = extract_demarcated_string(item, "---REASONING_START---", "---REASONING_END---")
                mab.add_correct_reasoning_to_gt(answer[1][i], reasoning)


        # Analyze why the llm got the question wrong
        # Can get test_id and correct_reasoning mapping via promt node data

        # print("mappings: ", prompt_node.train_mapping)
        
        list_of_testcases_for_analyze_cust = []
        for item in hardest_cases:
            to_add = {}
            to_add["_id"] = item
            to_add["prompt"] = prompt_node.train_mapping[item]["literal_prompt"]
            to_add["llm_answer"] = prompt_node.train_mapping[item]["answer"]
            to_add["ground_truth"] = train_set[train_set_lookup[item]]["answer"]
            to_add["correct_reasoning"] = mab.test_data[item]["reasoning"]
            list_of_testcases_for_analyze_cust.append(to_add)

        # print(list_of_testcases_for_analyze_cust)
        input_dict = {"prompt": "prompt", "llm_answer": "llm_answer", "ground_truth": "ground_truth", "correct_reasoning": "correct_reasoning"}
        answer = await batch_unified_call(student_llm, semaphore, list_of_testcases_for_analyze_cust, config["infer_hard_cases_prompt"], input_dict)
        for i, item in enumerate(answer[1]):
            prompt_node.train_hard_analysis[answer[1][i]] = extract_demarcated_string(answer[0][i], "---ANALYSIS_START---", "---ANALYSIS_END---")
        # print(prompt_node.train_hard_analysis)

        # @ DISTILLATION CALL
        input_for_distillation = []
        distillation = ""
        for k, v in prompt_node.train_hard_analysis.items():
            distillation += v + "\n"
        input_for_distillation.append({"_id": "distillation", "feedback_list": distillation, "original_prompt": prompt_node.prompt})
        input_dict = {"feedback_list": "feedback_list", "original_prompt": "original_prompt"}

        answer = await batch_unified_call(student_llm, semaphore, input_for_distillation, config["distill_patterns_from_hard_analysis"], input_dict)
        # print("\n\n answer: \n\n", answer)
        distilled_actionables = extract_demarcated_string(answer[0][0], "---DISTILLATION_START---", "---DISTILLATION_END---")
        # print("\n\n distillations \n\n", distilled_actionables)

        # @ UPDATE CALL
        input_for_prompt_update = []
        input_for_prompt_update.append({"_id": "generation", "original_prompt": prompt_node.prompt, "actionables": distilled_actionables})
        input_dict = {"original_prompt": "original_prompt", "actionables": "actionables"}
        answer = await batch_unified_call(generator_llm, semaphore, input_for_prompt_update, config["update_prompt"], input_dict)
        # print(answer[0][0])

        new_prompt = extract_demarcated_string(answer[0][0], "---PROMPT_START---", "---PROMPT_END---")
        print(new_prompt)
        if "---ANSWER_START---" not in new_prompt and "'---ANSWER_END---" not in new_prompt:
            new_prompt += "\nDemarcate your final answer to start with '---ANSWER_START---' and '---ANSWER_END---' verbatim, between which your actual answer will go."
        next_prompt = new_prompt
        print("hardest tests we sample for n-shots")        
        hardest_shots = mab.generate_k_shots_sample(train_set, train_set_lookup, 4)
        for i, item in enumerate(hardest_shots):
            print("example number ", i)
            print(item)
        # Checked ^^^

    

    # input_dict = {"prompt": "", "llm_answer": "", "ground_truth": "", "correct_reasoning": ""} # prompt, prompt, dataset
    # answer = None

    # Wrong Reasoning + Wrong Answer + Correct Reasoning + GT Answer >>> @ Analysis Call @ >>> Analysis on why Model Got Wrong

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(e)
        sys.exit(1)