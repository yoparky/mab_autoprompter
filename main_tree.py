import yaml
import asyncio
import sys
import heapq
import random
import json
import pandas as pd

import logging
import traceback

from llm_calls.llm_provider import create_llm_instance
from data.data_preprocessing import dataframe_to_list_of_dicts, squad_json_to_dataframe_from_file, split_data
from llm_calls.semaphore_call import batch_unified_call
from utils.prompt_node import PromptNode
from metrics.exact_match import exact_match
from metrics.f1_match import f1_match
from llm_calls.regex_extractor import extract_demarcated_string
from utils.mab import MAB

async def main():
    config_file_path = "./task_config.yaml"
    
    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)
    parameters_file_path = config["parameters_file_path"]
    
    with open(parameters_file_path, "r") as file:
        params = yaml.safe_load(file)
    dataset_file_path = config["dataset_file_path"]

    # must make sure datasets have a unique "_id" column
    dataset = dataframe_to_list_of_dicts(squad_json_to_dataframe_from_file(dataset_file_path))
    dataset = dataset[config["dataset_cut_start_index"]:config["dataset_cut_end_index"]]
    test_set, train_set, val_set = split_data(dataset)
    
    # llm setup
    student_llm = create_llm_instance(config["llm_provider"], config["llm_provider_model"], float(config["llm_provider_temperature"]))
    generator_llm = create_llm_instance(config["llm_generator"], config["llm_generator_model"], float(config["llm_generator_temperature"]))

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

    # set-up for original prompt object
    original_prompt = config["test_prompt"]
    next_prompt_params = "{}"
    prev_par = None

    prompt_node = PromptNode(original_prompt, prev_par)
    prompt_node.update_parameters(next_prompt_params)

    # generated prompts to validate
    to_validate = [prompt_node]

    # Main linear loop
    try:
        for _ in range(config["tree_generative_iteration"]):
            for generated_node in to_validate:
                # @ VALIDATION
                input_dict = {"context": "context", "question": "question"}
                answer = await batch_unified_call(student_llm, semaphore, val_set, generated_node.prompt, input_dict)

                generated_node.set_validation_data(answer, '---ANSWER_START---', '---ANSWER_END---')
                rewards = []
                for i, item in enumerate(generated_node.validation_answers):
                    reward = f1_match(val_set[val_set_lookup[generated_node.validation_ids[i]]], generated_node.validation_answers[i])
                    rewards.append(reward)
                val_score = round(sum(rewards)/len(rewards) if len(rewards) > 0 else 0.0, 4)
                generated_node.validation_rewards = rewards
                generated_node.validation_score = val_score
                generated_node.set_validation_mapping()
                # Update MAB parameter scores
                mab.update_params_result(generated_node.integrated_parameters, generated_node.validation_score)

                # Heap tracking best-performing items
                if len(node_heap) < config["heap_size"]:
                    heapq.heappush(node_heap, generated_node)
                else:
                    heapq.heappushpop(node_heap, generated_node)
            # At this point, all the nodes that were generated have updated validation scores and the heap is updated based on the validation values

            # We now must pick out the best unexplored node in the heap and start the expansion (generation) process
            ordered_list_of_best_prompts = heapq.nlargest(len(node_heap), node_heap)
            node_to_expand = None # the next node to start expansion
            for heap_prompt_node in ordered_list_of_best_prompts:
                if not heap_prompt_node.has_expanded:
                    node_to_expand = heap_prompt_node
                    node_to_expand.has_expanded = True
                    break
            if node_to_expand is None: break # END CONDITION - if all nodes in the heap has expanded, we stop the algo
            # at this point we have the node we have to train

            # @ TRAIN the selected node_to_expand
            input_dict = {"context": "context", "question": "question"}
            answer = await batch_unified_call(student_llm, semaphore, train_set, node_to_expand.prompt, input_dict)
            node_to_expand.set_train_data(answer, '---ANSWER_START---', '---ANSWER_END---')
            rewards = []
            for i, item in enumerate(node_to_expand.train_answers):
                reward = f1_match(train_set[train_set_lookup[node_to_expand.train_ids[i]]], node_to_expand.train_answers[i])
                mab.update_testcase_result(node_to_expand.train_ids[i], reward)
                rewards.append(reward)
            train_score = round(sum(rewards)/len(rewards) if len(rewards) > 0 else 0.0, 4)
            node_to_expand.train_rewards = rewards
            node_to_expand.train_score = train_score
            node_to_expand.set_train_mapping()

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
                answer = await batch_unified_call(generator_llm, semaphore, list_of_testcases_to_analyze, analysis_prompt, input_dict)
                for i, item in enumerate(answer[0]):
                    reasoning = extract_demarcated_string(item, "---REASONING_START---", "---REASONING_END---")
                    mab.add_correct_reasoning_to_gt(answer[1][i], reasoning)

            # @ INFER HARD QUESTION CALL
            list_of_testcases_for_analyze_cust = []
            for item in hardest_cases: # generate the dataset to add dynamically
                to_add = {}
                to_add["_id"] = item
                to_add["prompt"] = node_to_expand.train_mapping[item]["literal_prompt"]
                to_add["llm_answer"] = node_to_expand.train_mapping[item]["answer"]
                to_add["ground_truth"] = train_set[train_set_lookup[item]]["answer"]
                to_add["correct_reasoning"] = mab.test_data[item]["reasoning"]
                list_of_testcases_for_analyze_cust.append(to_add)
            input_dict = {"prompt": "prompt", "llm_answer": "llm_answer", "ground_truth": "ground_truth", "correct_reasoning": "correct_reasoning"}
            answer = await batch_unified_call(generator_llm, semaphore, list_of_testcases_for_analyze_cust, config["infer_hard_cases_prompt"], input_dict)
            for i, item in enumerate(answer[1]): # results are stored in node_to_expand
                node_to_expand.train_hard_analysis[answer[1][i]] = extract_demarcated_string(answer[0][i], "---ANALYSIS_START---", "---ANALYSIS_END---")

            # @ DISTILLATION CALL
            input_for_distillation = []
            distillation = ""
            for k, v in node_to_expand.train_hard_analysis.items():
                distillation += v + "\n"
            input_for_distillation.append({"_id": "distillation", "feedback_list": distillation, "original_prompt": node_to_expand.prompt})
            input_dict = {"feedback_list": "feedback_list", "original_prompt": "original_prompt"}
            answer = await batch_unified_call(generator_llm, semaphore, input_for_distillation, config["distill_patterns_from_hard_analysis"], input_dict)
            distilled_actionables = extract_demarcated_string(answer[0][0], "---DISTILLATION_START---", "---DISTILLATION_END---")

            # @ PARAMETER CALL
            input_for_parameter = []
            input_for_parameter.append({"_id": "param_selection","distilled_tips": distilled_actionables, "params": json.dumps(params), "active_parameters": node_to_expand.integrated_parameters}) # unify terminology
            input_dict = {"distilled_tips": "distilled_tips", "params": "params", "active_parameters": "active_parameters"}
            answer = await batch_unified_call(generator_llm, semaphore, input_for_parameter, config["parameter_selection_call"], input_dict)
            selected_parameters = extract_demarcated_string(answer[0][0], "---PARAMETER_START---", "---PARAMETER_END---")

            # @ RANK PARAMS BASED ON MAB, ONLY APPLY TOP J
            max_param_count_per_generation = config["max_param_sample_count"]
            json_selected_params = json.loads(selected_parameters)
            sorted_params_effective_first = []
            sorted_params_effective_first = mab._sample_sort_all_params()[::-1]
            params_to_apply = {}
            for i, value in enumerate(sorted_params_effective_first):
                if len(params_to_apply) >= max_param_count_per_generation: break
                if value in json_selected_params.keys():
                    params_to_apply[value] = json_selected_params[value]
            mab_filtered_parameters = json.dumps(params_to_apply) # %2 the mab filetered params based on top j performing params
            best_parameter = json.dumps({list(params_to_apply)[0]: params_to_apply[list(params_to_apply)[0]]}) if sorted_params_effective_first else "" # %3 the best param
            empty_parameter = "" # %4 empty params case

            # GENERATE K N-SHOTS FROM HARDEST ANALYZED TESTS
            hardest_shots = mab.generate_k_shots_sample(train_set, train_set_lookup, config["max_n_shots"])
            all_shots_str = "" # %5 all hard shots formatted to singular string
            for i, shot_str in enumerate(hardest_shots):
                all_shots_str += f"Example {i}:\n{shot_str}\n"

            zero_shots_str = "" # %6 zero shot
            one_shot_str = "" # %7 one shot
            if hardest_shots:
                one_shot_str += hardest_shots[0]
            
            random_int = random.randrange(len(hardest_shots))
            random_samples = random.sample(range(len(hardest_shots)), random_int)
            random_samples_shots_string = "" # %8 deterministic string based on order of hard samples
            for item in random_samples:
                random_samples_shots_string += f"Example:\n{hardest_shots[item]}\n"

            moderate_number = len(hardest_shots) // 2
            moderate_number_shots_str = "" # %9 a moderate (half of the length of n-shots) number of n-shots to try
            for i in range(moderate_number):
                moderate_number_shots_str += f"Example {i}:\n{hardest_shots[i]}\n"
            # pick strategies
            strategies = { # combination of parameters + n-shot example
                "mab_params_max_shot": {"_id": "generation", "original_prompt": node_to_expand.prompt, "actionables": mab_filtered_parameters, "n_shots": all_shots_str}, # 1
                "raw_params_zero_shot": {"_id": "generation", "original_prompt": node_to_expand.prompt, "actionables": selected_parameters, "n_shots": zero_shots_str}, # 2
                "best_params_one_shot": {"_id": "generation", "original_prompt": node_to_expand.prompt, "actionables": best_parameter, "n_shots": one_shot_str}, # 3
                "no_params_moderate_shot": {"_id": "generation", "original_prompt": node_to_expand.prompt, "actionables": empty_parameter, "n_shots": moderate_number_shots_str}, # 4
                # core set up to here
                "raw_params_max_shot": {"_id": "generation", "original_prompt": node_to_expand.prompt, "actionables": selected_parameters, "n_shots": all_shots_str}, # 5
                "mab_params_one_shot": {"_id": "generation", "original_prompt": node_to_expand.prompt, "actionables": mab_filtered_parameters, "n_shots": one_shot_str}, # 6
                "best_params_max_shot": {"_id": "generation", "original_prompt": node_to_expand.prompt, "actionables": best_parameter, "n_shots": all_shots_str}, # 7
                "no_params_one_shot": {"_id": "generation", "original_prompt": node_to_expand.prompt, "actionables": empty_parameter, "n_shots": one_shot_str}, # 8
                "mab_params_random_shot": {"_id": "generation", "original_prompt": node_to_expand.prompt, "actionables": mab_filtered_parameters, "n_shots": random_samples_shots_string}, # 9
                "raw_params_random_shot": {"_id": "generation", "original_prompt": node_to_expand.prompt, "actionables": selected_parameters, "n_shots": random_samples_shots_string}, # 10
                "best_params_random_shot": {"_id": "generation", "original_prompt": node_to_expand.prompt, "actionables": best_parameter, "n_shots": random_samples_shots_string}, # 11
                "no_params_random_shot": {"_id": "generation", "original_prompt": node_to_expand.prompt, "actionables": empty_parameter, "n_shots": random_samples_shots_string}, # 12
            }
            strategies_list = list(strategies.keys())
            core_strategies_keys_list = ["mab_params_max_shot", "raw_params_zero_shot", "best_params_one_shot", "no_params_moderate_shot"]
            picked_strategies = []
            generation_count_per_expansion = config["children_generation_per_node_expansion"]
            if generation_count_per_expansion <= 0:
                picked_strategies = core_strategies_keys_list
            elif generation_count_per_expansion <= len(core_strategies_keys_list):
                for i in range(generation_count_per_expansion):
                    picked_strategies.append(core_strategies_keys_list[i])
            elif generation_count_per_expansion <= len(strategies_list):
                for i in range(generation_count_per_expansion):
                    picked_strategies.append(strategies_list[i])
            else: # note we can also random.choices(pop, k=num_samples). Currently do deterministic
                extra_generation = generation_count_per_expansion - len(strategies_list)
                picked_strategies.extend(strategies_list)
                for _ in range(extra_generation):
                    index_of_strat_to_add = random.randint(0, len(strategies_list) - 1)
                    picked_strategies.append(strategies_list[index_of_strat_to_add])
            # @ GENERATE NEW PROMPT CALL
            # Strategy selection complete, generate list based on selected strategies
            to_validate = [] # reset to_validate
            input_for_prompt_update = []
            for key_str in picked_strategies:
                input_for_prompt_update.append(strategies[key_str])
            input_dict = {"original_prompt": "original_prompt", "actionables": "actionables", "n_shots": "n_shots"}
            answer = await batch_unified_call(generator_llm, semaphore, input_for_prompt_update, config["update_prompt"], input_dict)
            for i, call_input in enumerate(input_for_prompt_update):
                param_string = call_input["actionables"]
                new_prompt_str = extract_demarcated_string(answer[0][i], "---PROMPT_START---", "---PROMPT_END---")
                if "---ANSWER_START---" not in new_prompt_str and "'---ANSWER_END---" not in new_prompt_str:
                    new_prompt_str += "\nDemarcate your final answer with '---ANSWER_START---' and '---ANSWER_END---' verbatim, between which your actual answer will go."
                generated_prompt_obj = PromptNode(new_prompt_str, node_to_expand)
                generated_prompt_obj.update_parameters(param_string)
                node_to_expand.children_ids.append(generated_prompt_obj.id)
                to_validate.append(generated_prompt_obj)
    except Exception as e:
        outer_traceback = traceback.format_exc()
        logging.error(
            f"Type of exception: {type(e).__name__}\n"
            f"Exception details: {e}\n"
            f"Full Traceback from outer catch:\n{outer_traceback}"
        )

    # TEST CALL ON BEST VALIDATION-PERFORMANCE PROMPTS IN HEAP
    ordered_list_of_best_prompts = heapq.nlargest(len(node_heap), node_heap)
    best_prompt = ordered_list_of_best_prompts[0]
    print("\nbest prompt validation: ", best_prompt)
    print("score: ", best_prompt.validation_score,)

    # @ TEST CALL for baseline prompt
    input_dict = {"context": "context", "question": "question"}
    answer = await batch_unified_call(student_llm, semaphore, test_set, prompt_node.prompt, input_dict)
    prompt_node.set_test_data(answer, '---ANSWER_START---', '---ANSWER_END---')
    rewards = []
    for i, item in enumerate(prompt_node.test_answers):
        reward = f1_match(test_set[test_set_lookup[prompt_node.test_ids[i]]], prompt_node.test_answers[i])
        rewards.append(reward)
    test_score = round(sum(rewards)/len(rewards) if len(rewards) > 0 else 0.0, 4)
    prompt_node.test_rewards = rewards
    prompt_node.test_score = test_score
    prompt_node.set_test_mapping()

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
    
    print("best prompt test: ", ordered_list_of_best_prompts[-1].prompt)
    print("\nintegrated parameters: ", ordered_list_of_best_prompts[-1].integrated_parameters)

    # Write logs
    list_for_df = []
    for node in node_heap:
        dict_of_node = node.to_dict()
        list_for_df.append(dict_of_node)
    
    df_nodes = pd.DataFrame(list_for_df)

    csv_file_path = "./results/results_dataframe.csv"
    try:
        df_nodes.to_csv(csv_file_path, index=False, encoding='utf-8')
    except Exception as e:
        print(f"error writing csv: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(e)
        sys.exit(1)