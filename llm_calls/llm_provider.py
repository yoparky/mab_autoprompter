import os
import sys
import logging
from typing import Dict, Tuple, Optional, List

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
# from langchain_core.callbacks import BaseCallbackHandler

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_community.llms import VLLM

API_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "ollama": "OLLAMA_API_KEY",
    "vllm": "VLLM_API_KEY",
    "url": "URL_API_KEY",
}

def create_llm_instance(
    provider: str,
    model: str,
    temperature: float = 0.3,
    base_url: str = "",
) -> BaseChatModel:
    
    load_dotenv()
    llm = None
    api_key = os.getenv(API_MAP[provider])
    if provider not in API_MAP.keys() or api_key is None: return llm

    if provider == "openai":
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            cache=False,
        )
    elif provider == "anthropic":
        llm = ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=api_key,
            cache=False,
        )
    elif provider == "google":
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key,
            convert_system_message_to_human=True,
        )
    elif provider == "ollama":
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        llm = ChatOllama(
            model=model,
            temperature=temperature,
            base_url=ollama_base_url,
            # enable_thinking=False, top_k=config.get('ollama_top_k'), top_p=config.get('ollama_top_p'),
        )
    elif provider == "vllm":
        llm = VLLM(
            model=model,
            trust_remote_code=True, # take care
            temperature=temperature,
            # top_k=1, top_p=0.95, max_new_tokens=128,
        )
    elif provider == "url": # ChatOpenAI API
        base_url = base_url
        llm = ChatOpenAI(
            model=model, base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            # model_kwargs={"reasoning_effort": "medium"},
        )
    return llm
