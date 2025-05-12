import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel 

async def unified_call(llm, semaphore, testcase, prompt, input_dict):
    async with semaphore:
        chat_prompt = ChatPromptTemplate.from_template(prompt, template_format="jinja2")
        chain = chat_prompt | llm
        try:
            processed_input_dict = {prompt_key: testcase[testcase_key] for prompt_key, testcase_key in input_dict.items()}
            used_prompt_literal = await chat_prompt.ainvoke(processed_input_dict)
            used_prompt_literal = used_prompt_literal.to_string()
            response = await chain.ainvoke(processed_input_dict)
            return response, testcase["_id"], used_prompt_literal
        except Exception as e: 
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
        input_tokens.append(item[0].response_metadata.get("token_usage").get("prompt_tokens"))
        output_tokens.append(item[0].response_metadata.get("token_usage").get("completion_tokens"))
        ids.append(item[1])
        literal_prompt.append(item[2])

    return answers, ids, input_tokens, output_tokens, literal_prompt
