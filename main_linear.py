import yaml
import asyncio
import sys
import heapq
import random
import json

from llm_provider import create_llm_instance
from data_preprocessing import dataframe_to_list_of_dicts, squad_json_to_dataframe_from_file, split_data
from semaphore_call import batch_unified_call
from prompt_node import PromptNode
from metrics.exact_match import exact_match
from metrics.f1_match import f1_match
from regex_extractor import extract_demarcated_string
from mab import MAB
# REMEMBER TO UNIFY CALLS FOR TREES

async def main():
    config_file_path = "./task_config.yaml"
    squad_file_path = "./squad-train-v2.0.json"
    parameters_file_path = "./prompt_parameters_v1.yaml"

    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)
    with open(parameters_file_path, "r") as file:
        params = yaml.safe_load(file) 
    
    # must make sure datasets have a unique "_id" column
    dataset = dataframe_to_list_of_dicts(squad_json_to_dataframe_from_file(squad_file_path))
    dataset = dataset[:400]
    test_set, train_set, val_set = split_data(dataset)
    
    # llm setup
    student_llm = create_llm_instance(config["llm_provider"], config["llm_provider_model"], float(config["llm_provider_temperature"]))
    generator_llm = create_llm_instance(config["llm_provider"], config["llm_generator_model"], float(config["llm_generator_temperature"]))

    # concurrency
    max_concurrent_calls = config["max_concurrent_calls"]
    semaphore = asyncio.Semaphore(max_concurrent_calls)

    # heap
    node_heap = []
    heapq.heapify(node_heap)

    # Test case logic start. Can use same logic for test set
    val_set_lookup = {}
    for i, item in enumerate(val_set):
        val_set_lookup[item["_id"]] = i
    train_set_lookup = {}
    for i, item in enumerate(train_set):
        train_set_lookup[item["_id"]] = i
    test_set_lookup = {}
    for i, item in enumerate(test_set):
        test_set_lookup[item["_id"]] = i

    # MAB
    mab = MAB(train_set)
    mab.initialize_params_data(params)

    # hand-off values for loop
    next_prompt = config["test_prompt"]
    next_prompt_params = "{}"
    prev_prompt = None

    # Main linear loop
    for k in range(config["loop_iteration_count"]):
        # prompt
        prompt_node = PromptNode(next_prompt, prev_prompt)
        prompt_node.update_parameters(next_prompt_params)

        # @ VALIDATION
        input_dict = {"context": "context", "question": "question"}
        answer = await batch_unified_call(student_llm, semaphore, val_set, prompt_node.prompt, input_dict)
        prompt_node.set_validation_data(answer, '---ANSWER_START---', '---ANSWER_END---')
        rewards = []
        for i, item in enumerate(prompt_node.validation_answers):
            reward = f1_match(val_set[val_set_lookup[prompt_node.validation_ids[i]]], prompt_node.validation_answers[i])
            rewards.append(reward)
        val_score = round(sum(rewards)/len(rewards) if len(rewards) > 0 else 0.0, 4)
        prompt_node.validation_rewards = rewards
        prompt_node.validation_score = val_score
        prompt_node.set_validation_mapping()
        # Update MAB parameter scores
        mab.update_params_result(prompt_node.integrated_parameters, prompt_node.validation_score)
        # BASIC LOGGING
        print(prompt_node.id)
        print("validation")
        # print(prompt_node.validation_rewards)
        print(prompt_node.validation_score)
        # BASIC LOGGING END

        # Heap tracking best-performing items
        if len(node_heap) < config["heap_size"]:
            heapq.heappush(node_heap, prompt_node)
        else:
            heapq.heappushpop(node_heap, prompt_node)
        
        # @ TRAIN
        input_dict = {"context": "context", "question": "question"}
        answer = await batch_unified_call(student_llm, semaphore, train_set, prompt_node.prompt, input_dict)
        prompt_node.set_train_data(answer, '---ANSWER_START---', '---ANSWER_END---')
        rewards = []
        for i, item in enumerate(prompt_node.train_answers):
            reward = f1_match(train_set[train_set_lookup[prompt_node.train_ids[i]]], prompt_node.train_answers[i])
            mab.update_testcase_result(prompt_node.train_ids[i], reward)
            rewards.append(reward)
        train_score = round(sum(rewards)/len(rewards) if len(rewards) > 0 else 0.0, 4)
        prompt_node.train_rewards = rewards
        prompt_node.train_score = train_score
        prompt_node.set_train_mapping()
        # BASIC LOGGING
        print("train")
        # print(prompt_node.train_rewards)
        print(prompt_node.train_score)
        # BASIC LOGGING END

        # CORRECT REASONING INFERENCE FOR PREVIOUSLY UNPROCESSED HARD QUESTIONS
        hardest_cases = mab.sample_k_hardest(config["hard_sample_count"]) # list of str, [_id]
        list_of_testcases_to_analyze = []
        for item in train_set:
            if item['_id'] in hardest_cases and not mab.has_correct_reasoning_to_gt(item['_id']):
                list_of_testcases_to_analyze.append(item)
        # Retrieve unpopulated analyses for hard questions that have not been analyzed
        if list_of_testcases_to_analyze:
            analysis_prompt = config["analyze_correct_reasoning_prompt"]
            input_dict = {"context": "context", "question": "question", "answer": "answer"}
            answer = await batch_unified_call(student_llm, semaphore, list_of_testcases_to_analyze, analysis_prompt, input_dict)
            for i, item in enumerate(answer[0]):
                reasoning = extract_demarcated_string(item, "---REASONING_START---", "---REASONING_END---")
                mab.add_correct_reasoning_to_gt(answer[1][i], reasoning)

        # @ INFER HARD QUESTION CALL
        list_of_testcases_for_analyze_cust = []
        for item in hardest_cases: # generate the dataset to add dynamically
            to_add = {}
            to_add["_id"] = item
            to_add["prompt"] = prompt_node.train_mapping[item]["literal_prompt"]
            to_add["llm_answer"] = prompt_node.train_mapping[item]["answer"]
            to_add["ground_truth"] = train_set[train_set_lookup[item]]["answer"]
            to_add["correct_reasoning"] = mab.test_data[item]["reasoning"]
            list_of_testcases_for_analyze_cust.append(to_add)
        input_dict = {"prompt": "prompt", "llm_answer": "llm_answer", "ground_truth": "ground_truth", "correct_reasoning": "correct_reasoning"}
        answer = await batch_unified_call(student_llm, semaphore, list_of_testcases_for_analyze_cust, config["infer_hard_cases_prompt"], input_dict)
        for i, item in enumerate(answer[1]): # results are stored in prompt_node
            prompt_node.train_hard_analysis[answer[1][i]] = extract_demarcated_string(answer[0][i], "---ANALYSIS_START---", "---ANALYSIS_END---")

        # @ DISTILLATION CALL
        input_for_distillation = []
        distillation = ""
        for k, v in prompt_node.train_hard_analysis.items():
            distillation += v + "\n"
        input_for_distillation.append({"_id": "distillation", "feedback_list": distillation, "original_prompt": prompt_node.prompt})
        input_dict = {"feedback_list": "feedback_list", "original_prompt": "original_prompt"}
        answer = await batch_unified_call(student_llm, semaphore, input_for_distillation, config["distill_patterns_from_hard_analysis"], input_dict)
        distilled_actionables = extract_demarcated_string(answer[0][0], "---DISTILLATION_START---", "---DISTILLATION_END---")
        # BASIC LOGGING
        print(">>>>>> demo distillation >>>>>>")
        print(distilled_actionables, "\n\n")
        # BASIC LOGGING END

        # @ PARAMETER CALL
        input_for_parameter = []
        input_for_parameter.append({"_id": "param_selection","distilled_tips": distilled_actionables, "params": json.dumps(params), "active_parameters": prompt_node.integrated_parameters}) # unify terminology
        input_dict = {"distilled_tips": "distilled_tips", "params": "params", "active_parameters": "active_parameters"}
        answer = await batch_unified_call(student_llm, semaphore, input_for_parameter, config["parameter_selection_call"], input_dict)
        selected_parameters = extract_demarcated_string(answer[0][0], "---PARAMETER_START---", "---PARAMETER_END---")
        # BASIC LOGGING
        print(">>>>>> demo param selection >>>>>>")
        print(selected_parameters, "\n\n")
        # BASIC LOGGING END

        # @ RANK PARAMS BASED ON MAB, ONLY APPLY TOP J
        max_param_count_per_generation = config["max_param_sample_count"]
        json_selected_params = json.loads(selected_parameters)
        sorted_params_effective_first = reversed(mab._sample_sort_all_params())
        params_to_apply = {}
        for i, value in enumerate(sorted_params_effective_first):
            if len(params_to_apply) >= max_param_count_per_generation: break
            if value in json_selected_params.keys():
                params_to_apply[value] = json_selected_params[value]
        selected_parameters = json.dumps(params_to_apply)
        # BASIC LOGGING
        print("seleted MAB params: ", params_to_apply)
        # BASIC LOGGING END
        
        # GENERATE K N-SHOTS FROM HARDEST ANALYZED TESTS
        hardest_shots = mab.generate_k_shots_sample(train_set, train_set_lookup, config["max_n_shots"])
        random_number = random.randint(0, len(hardest_shots))
        shots_string = ""
        for i in range(random_number):
            shots_string = shots_string + "\n" + hardest_shots[i]
        print(">>>>>>>generated n-shots>>>>>>\n", shots_string)

        # @ GENERATE NEW PROMPT CALL
        input_for_prompt_update = []
        input_for_prompt_update.append({"_id": "generation", "original_prompt": prompt_node.prompt, "actionables": selected_parameters, "n_shots": shots_string})
        input_dict = {"original_prompt": "original_prompt", "actionables": "actionables", "n_shots": "n_shots"}
        answer = await batch_unified_call(generator_llm, semaphore, input_for_prompt_update, config["update_prompt"], input_dict)
        new_prompt = extract_demarcated_string(answer[0][0], "---PROMPT_START---", "---PROMPT_END---")
        if "---ANSWER_START---" not in new_prompt and "'---ANSWER_END---" not in new_prompt:
            new_prompt += "\nDemarcate your final answer to start with '---ANSWER_START---' and '---ANSWER_END---' verbatim, between which your actual answer will go."
        
        print(">>>>>>new prompt>>>>>>\n", new_prompt)
        # hand-off update
        next_prompt = new_prompt
        prev_prompt = prompt_node
        next_prompt_params = selected_parameters


    # TEST CALL ON BEST VALIDATION-PERFORMANCE PROMPTS IN HEAP
    ordered_list_of_best_prompts = heapq.nlargest(len(node_heap), node_heap)
    best_prompt = ordered_list_of_best_prompts[0]
    print("\nbest prompt validation\n", best_prompt)
    print("score: ", best_prompt.validation_score,)
    # @ TEST CALL
    for curr_node in ordered_list_of_best_prompts:
        input_dict = {"context": "context", "question": "question"}
        answer = await batch_unified_call(student_llm, semaphore, test_set, curr_node.prompt, input_dict)
        curr_node.set_test_data(answer, '---ANSWER_START---', '---ANSWER_END---')
        rewards = []
        for i, item in enumerate(curr_node.test_answers):
            reward = f1_match(test_set[test_set_lookup[curr_node.test_ids[i]]], curr_node.test_answers[i])
            rewards.append(reward)
        test_score = round(sum(rewards)/len(rewards) if len(rewards) > 0 else 0.0, 4)
        curr_node.test_rewards = rewards
        curr_node.test_score = test_score
        curr_node.set_test_mapping()

        # BASIC LOGGING
        print("id: ", curr_node.id)
        print("test")
        print(curr_node.test_score)
        # BASIC LOGGING END
    # order based on test_score
    ordered_list_of_best_prompts.sort(key = lambda x: x.test_score)
    for node in ordered_list_of_best_prompts:
        print(node.id)
        print(node.test_score)
    
    print("best prompt: ", ordered_list_of_best_prompts[-1].prompt)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(e)
        sys.exit(1)