import yaml
import asyncio
import sys
from llm_calls.llm_provider import create_llm_instance
from data.data_preprocessing import dataframe_to_list_of_dicts, squad_json_to_dataframe_from_file, split_data, hpqa_filepath_to_list_of_testcases, hf_dataset_to_list_of_dict, jsonl_filepath_to_list_of_testcases
from llm_calls.semaphore_call import batch_unified_call
from utils.prompt_node import PromptNode
from metrics.exact_match import exact_match, exact_match_gsm8k
from metrics.f1_match import f1_match

async def main():

    config_file_path = "./evaluation_script_config.yaml"
    
    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)
    llm_prompt_config_path = config["llm_prompt_config_path"]
    dataset_file_path = config["dataset_file_path"]

    with open(llm_prompt_config_path, "r") as file:
        prompt_config = yaml.safe_load(file)

    # DATA PREP
    # must make sure datasets have a unique "_id" column
    # dataset = dataframe_to_list_of_dicts(squad_json_to_dataframe_from_file(dataset_file_path)) # SQuAD
    # dataset = hpqa_filepath_to_list_of_testcases(dataset_file_path) # HPQA
    dataset = jsonl_filepath_to_list_of_testcases(dataset_file_path)

    # Data split
    dataset = dataset[config["dataset_cut_start_index"]:config["dataset_cut_end_index"]]
    test_set, _, _ = split_data(dataset, config["test_ratio"], config["train_ratio"], config["val_ratio"])

    # llm setup
    student_llm = create_llm_instance(config["llm_provider"], config["llm_provider_model"], float(config["llm_provider_temperature"]))

    # concurrency
    max_concurrent_calls = config["max_concurrent_calls"]
    semaphore = asyncio.Semaphore(max_concurrent_calls)

    # Test case logic start. Can use same logic for test set
    test_set_lookup = {}
    for i, item in enumerate(test_set):
        test_set_lookup[item["_id"]] = i

    # set-up for original prompt object
    original_prompt = prompt_config["test_prompt"]
    prompt_node = PromptNode(original_prompt, None)

    # @ TEST CALL for baseline prompt
    answer = await batch_unified_call(student_llm, semaphore, test_set, prompt_node.prompt, prompt_config['test_prompt_input_dict'])
    prompt_node.set_test_data(answer, '---ANSWER_START---', '---ANSWER_END---')
    rewards = []
    for i, item in enumerate(prompt_node.test_answers):
        reward = exact_match_gsm8k(test_set[test_set_lookup[prompt_node.test_ids[i]]], prompt_node.test_answers[i])
        rewards.append(reward)
    test_score = round(sum(rewards)/len(rewards) if len(rewards) > 0 else 0.0, 4)
    prompt_node.test_rewards = rewards
    prompt_node.test_score = test_score
    prompt_node.set_test_mapping()

    print("Prompt:\n", prompt_node.prompt)
    print("Score: ", test_score)
    # print("Reward:\n", prompt_node.test_rewards)
    # print("Answer:\n", prompt_node.test_answers)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(e)
        sys.exit(1)