from langchain_openrouter import ChatOpenRouter
from langchain_ollama import ChatOllama
from settings.settings import settings


def get_llm():
    return ChatOllama(
        model="gpt-oss:120b-cloud",
        # base_url="https://openrouter.ai/api/v1",
        # api_key=settings.LLM_API_KEY,
        temperature=0,
    )
    # return ChatOpenRouter(
    #         model="openai/gpt-oss-120b:free",
    #         # base_url="https://openrouter.ai/api/v1",
    #         api_key=settings.LLM_API_KEY,
    #         temperature=0,
    #     )

