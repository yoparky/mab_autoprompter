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
from data.data_preprocessing import dataframe_to_list_of_dicts, squad_json_to_dataframe_from_file, split_data, hpqa_filepath_to_list_of_testcases, hf_dataset_to_list_of_dict, jsonl_filepath_to_list_of_testcases
from llm_calls.semaphore_call import batch_unified_call
from utils.prompt_node import PromptNode
from metrics.exact_match import exact_match, exact_match_gsm8k
from metrics.f1_match import f1_match
from llm_calls.regex_extractor import extract_demarcated_string

# Implementation of OPRO as described in:
# https://arxiv.org/abs/2309.03409

async def main():
    config_file_path = "./task_config.yaml"
    
    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)
    llm_prompt_config_path = config["llm_prompt_config_path"]

    dataset_file_path = config["dataset_file_path"]

    with open(llm_prompt_config_path, "r") as file:
        prompt_config = yaml.safe_load(file)

    # DATA PREP
    dataset = jsonl_filepath_to_list_of_testcases(dataset_file_path)

    # Data split
    dataset = dataset[config["dataset_cut_start_index"]:config["dataset_cut_end_index"]]
    test_set, train_set, val_set = split_data(dataset, config["test_ratio"], config["train_ratio"], config["val_ratio"])

    # llm setup
    student_llm = create_llm_instance(config["llm_provider"], config["llm_provider_model"], float(config["llm_provider_temperature"]))
    advanced_generator_llm = create_llm_instance(config["llm_advanced_generator"], config["llm_advanced_generator_model"], float(config["llm_advanced_generator_temperature"]))

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

    # set-up for original prompt object
    original_prompt = prompt_config["test_prompt"]
    prev_par = None
    prompt_node = PromptNode(original_prompt, prev_par)
    heapq.heappush(node_heap, prompt_node)

    # generated prompts to validate
    to_validate = [prompt_node]

    for _ in range(config["opro_iteration"]):
        try:
            for generated_node in to_validate:
                # @ VALIDATION
                answer = await batch_unified_call(student_llm, semaphore, val_set, generated_node.prompt, prompt_config['test_prompt_input_dict'])
                generated_node.set_validation_data(answer, '|', '|')
                rewards = []
                for i, item in enumerate(generated_node.validation_answers):
                    reward = exact_match_gsm8k(val_set[val_set_lookup[generated_node.validation_ids[i]]], generated_node.validation_answers[i])
                    rewards.append(reward)
                val_score = round(sum(rewards)/len(rewards) if len(rewards) > 0 else 0.0, 4)
                generated_node.validation_rewards = rewards
                generated_node.validation_score = val_score
                generated_node.set_validation_mapping()

                # Heap tracking best-performing items
                if len(node_heap) < config["heap_size"]:
                    heapq.heappush(node_heap, generated_node)
                else:
                    heapq.heappushpop(node_heap, generated_node)
                # Peek and check the biggest number

            # Combine the prompts
            ascending_ordered_list_of_best_prompts = heapq.nsmallest(len(node_heap), node_heap)
            best_performing_prompts_string = ""
            for prompt_node in ascending_ordered_list_of_best_prompts:
                prompt_str = f"text:\n{prompt_node.prompt}\nscore:\n{round((prompt_node.validation_score * 100), 2)}%\n"
                best_performing_prompts_string += prompt_str
            # Combine the exemplars
            sampled_exemplars = []
            if config["opro_exemplar_count"] > len(train_set): sampled_exemplars = train_set
            else: sampled_exemplars = random.sample(train_set, k=config["opro_exemplar_count"])
            exemplar_string = ""
            for test_case in sampled_exemplars:
                test_case_string = f"Context: {test_case.context}\nQ: {test_case.question}\nA: <INS>\noutput:\n{test_case.ans}\n"
                exemplar_string += test_case_string
            meta_prompt_str = f"""I have some texts along with their corresponding scores. The texts are arranged in ascending order based on their scores, where higher scores indicate better quality.\n\n{best_performing_prompts_string}\nThe following exemplars show how to apply your text: you replace <INS> in each input with your text, then read the input and give an output. We say your output is wrong if your output is different from the given output, and we say your output is correct if they are the same.\n\n{exemplar_string}\n\nWrite your new text that is different from the old ones and has a score as high as possible. Write the text in square brackets."""

            # @ GENERATION CALL
            # Strategy selection complete, generate list based on selected strategies
            impromptu_list_for_generation = []
            for item in range(int(config["opro_generation_per_iteration"])):
                to_add = {}
                to_add["_id"] = "generative_call"
                impromptu_list_for_generation.append(to_add)
            to_validate = []
            generated_nodes = []
            answer = await batch_unified_call(advanced_generator_llm, semaphore, impromptu_list_for_generation, meta_prompt_str, prompt_config['update_input_dict'])
            for i, item in enumerate(answer[0]):
                generated_prompt_node = PromptNode(extract_demarcated_string(answer[0][i], "|", "|"))
                generated_nodes.append(generated_prompt_node)
            to_validate.extend(generated_nodes)

        except Exception as e:
            outer_traceback = traceback.format_exc()
            logging.error(
                f"Type of exception: {type(e).__name__}\n"
                f"Exception details: {e}\n"
                f"Full Traceback from outer catch:\n{outer_traceback}"
            )
            continue
    
    # TEST CALL ON BEST VALIDATION-PERFORMANCE PROMPTS IN HEAP
    ordered_list_of_best_prompts = heapq.nlargest(len(node_heap), node_heap)
    best_prompt = ordered_list_of_best_prompts[0]
    print("\nbest prompt validation: ", best_prompt)
    print("score: ", best_prompt.validation_score,)

    # @ TEST CALL for baseline prompt
    answer = await batch_unified_call(student_llm, semaphore, test_set, prompt_node.prompt, prompt_config['test_prompt_input_dict'])
    prompt_node.set_test_data(answer, '|', '|')
    rewards = []
    for i, item in enumerate(prompt_node.test_answers):
        reward = exact_match_gsm8k(test_set[test_set_lookup[prompt_node.test_ids[i]]], prompt_node.test_answers[i])
        rewards.append(reward)
    test_score = round(sum(rewards)/len(rewards) if len(rewards) > 0 else 0.0, 4)
    prompt_node.test_rewards = rewards
    prompt_node.test_score = test_score
    prompt_node.set_test_mapping()

    # @ TEST CALL
    for curr_node in ordered_list_of_best_prompts:
        answer = await batch_unified_call(student_llm, semaphore, test_set, curr_node.prompt, prompt_config['test_prompt_input_dict'])
        curr_node.set_test_data(answer, '|', '|')
        rewards = []
        for i, item in enumerate(curr_node.test_answers):
            reward = exact_match_gsm8k(test_set[test_set_lookup[curr_node.test_ids[i]]], curr_node.test_answers[i])
            rewards.append(reward)
        test_score = round(sum(rewards)/len(rewards) if len(rewards) > 0 else 0.0, 4)
        curr_node.test_rewards = rewards
        curr_node.test_score = test_score
        curr_node.set_test_mapping()
    
    print("best prompt test: ", ordered_list_of_best_prompts[0].prompt)

    # Write logs
    list_for_df = []
    for node in node_heap:
        dict_of_node = node.to_dict()
        list_for_df.append(dict_of_node)
    
    df_nodes = pd.DataFrame(list_for_df)

    csv_file_path = config["output_file_path"]
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