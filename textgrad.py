import os
import textgrad as tg
from textgrad.engine.openai import ChatOpenAI
import pandas as pd

# LLM Setup
os.environ['OPENAI_API_KEY'] = "vllm-compatible-server"
os.environ['OPENAI_BASE_URL'] = "http://localhost:1234/v1" # config this
engine = ChatOpenAI(model_string='<your-model-name>')
tg.set_backwards_engine(engine, override=True)

# dataset
decisions_dataset = [
    {"question": "Tackle from behind", "answer": "Yellow card"},
    {"question": "Hand ball in the penalty area", "answer": "Penalty"}
]
decisions_df = pd.DataFrame(decisions_dataset)
decisions_df.answer = decisions_df.answer.str.lower()

# prompt
system_prompt = tg.Variable(value="""
You are a football analyst. I will give you a situation in Football, and you must choose from the following:
['yellow card', 'penalty', 'red card', 'corner kick', 'var', 'warning'].
Your output should exactly match the given values. No extra explanation, just the category string.""",
requires_grad = True,
role_description="system prompt"
)

# llm = tg.get_engine(engine_name="gpt-3.5-turbo")
model = tg.BlackboxLLM(engine, system_prompt)

format_string = """
LLM Prompt: {system_prompt}
Query: {query}
Prediction: {pred}
Ground Truth: {target}
Evaluation: {eval}
"""

loss_system_prompt = tg.Variable("""
Your job is to provide feedback to a LLM classifier. You will get the question,
the LLM generated answer as well as the intended ground truth label.
The LLM output should EXACTLY match the ground truth target, and the eval Evaluation be True.
You must provide concise feedback to correct the response.
""",
role_description="System prompt to provide feedback",
requires_grad=False
)

fields = {"system_prompt": None, "query": None, "pred": None, "target": None, "eval": None}

formatted_llm_call = tg.autograd.FormattedLLMCall(engine=tg.get_engine("gpt-4o-mini"), # or just the local 'engine'
                                                  format_string=format_string,
                                                  fields=fields,
                                                  systemp_prompt=loss_system_prompt)
optimizer = tg.TGD([system_prompt])

def loss_fn(system_prompt, query, pred, target, eval):
    return formatted_llm_call(
        inputs = {"system_prompt": system_prompt, "query": query, "pred": pred, "target": target, "eval": eval}
    )


def finetune_prompt(df):
    losses = []
    for idx in range(len(df)):
        question_str = decisions_df.iloc[idx].question
        target = decisions_df.iloc[idx].answer
        question = tg.Variable(question_str, requires_grad=False, role_description="Football situation query")
        target = tg.Variable(target, requires_grad=False, role_description="Ground Truth Answer")
        pred = model(question)

        if (pred.value.lower() == target.value.lower()):
            eval_str = "Correct as prediction exactly matches target"
        else:
            eval_str = f"Incorrect as prediction doesn't exactly match target. LLM Response: {pred.value}. Correct output: {target.value}"

        eval = tg.Variable(eval_str, requires_grad=False, role_description="Evaluation")

        losses.append(loss_fn(
            system_prompt=system_prompt,
            query=question,
            pred=pred,
            target=target,
            eval=eval,
        ))

        if len(losses) == 10:
            total_loss = tg.sum(losses)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # print("Eval: ", evaluation_score(decisions_df.iloc[:20]))
            losses = []

    return model

model = finetune_prompt(decisions_df.iloc[:60])

print(system_prompt)