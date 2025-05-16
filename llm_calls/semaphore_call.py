import asyncio

# import logging
# import traceback

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel 

async def unified_call(llm, semaphore, testcase, prompt, input_dict):
    async with semaphore:
        chat_prompt = ChatPromptTemplate.from_template(prompt, template_format="jinja2")
        chain = chat_prompt | llm
        try:
            processed_input_dict = {prompt_key: testcase[testcase_key] for prompt_key, testcase_key in input_dict.items() if testcase_key in testcase.keys()}
            used_prompt_literal = await chat_prompt.ainvoke(processed_input_dict)
            used_prompt_literal = used_prompt_literal.to_string()
            response = await chain.ainvoke(processed_input_dict)
            return response, testcase["_id"], used_prompt_literal
        except Exception as e: 
            # error_type = type(e).__name__
            # error_message = str(e)
            # full_traceback = traceback.format_exc()
            # logging.error(
            #     f"Error in unified_call for testcase_id '{testcase.get('_id', 'UnknownID')}':\n"
            #     f"Type: {error_type}\n"
            #     f"Message: {error_message}\n"
            #     f"Traceback:\n{full_traceback}"
            # )
            return None, testcase["_id"], None

async def batch_unified_call(llm, semaphore, testcases, prompt, input_dict):
    tasks=[]
    for i, _ in enumerate(testcases):
        task = asyncio.create_task(unified_call(llm, semaphore, testcases[i], prompt, input_dict))
        tasks.append(task)
    result = await asyncio.gather(*tasks)
    answers, ids, input_tokens, output_tokens, literal_prompt = [], [], [], [], []
    for item in result:
        if item[0] is None: continue
        answers.append(item[0].content)
        input_tokens.append(item[0].usage_metadata.get("input_tokens"))
        output_tokens.append(item[0].usage_metadata.get("output_tokens"))
        ids.append(item[1])
        literal_prompt.append(item[2])

    return answers, ids, input_tokens, output_tokens, literal_prompt
