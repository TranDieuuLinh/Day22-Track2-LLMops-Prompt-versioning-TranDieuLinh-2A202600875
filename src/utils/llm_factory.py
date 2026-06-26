"""
Factory tạo LLM và Embeddings cho 5 providers: openai, gemini, anthropic, ollama, openrouter.

Cách dùng:
    from utils.llm_factory import get_llm, get_embeddings

    llm        = get_llm()            # dùng PROVIDER từ .env
    embeddings = get_embeddings()     # dùng PROVIDER từ .env

    llm_gemini = get_llm("gemini")    # chỉ định provider cụ thể
"""
import config
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


"""
Factory function để tạo LLM và Embeddings dựa trên PROVIDER từ config.py
"""


def get_llm():
    """
    Trả về LLM object dựa trên PROVIDER trong config.
    """
    if config.PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=config.OPENAI_API_KEY,
            model=config.OPENAI_MODEL,
            temperature=0,
            base_url=config.OPENAI_BASE_URL if config.OPENAI_BASE_URL else None,
        )

    elif config.PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            api_key=config.GOOGLE_API_KEY,
            model=config.GEMINI_MODEL,
            temperature=0,
        )

    elif config.PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            api_key=config.ANTHROPIC_API_KEY,
            model=config.ANTHROPIC_MODEL,
            temperature=0,
        )

    elif config.PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=config.OLLAMA_BASE_URL,
            model=config.OLLAMA_MODEL,
            temperature=0,
        )

    elif config.PROVIDER == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            model=config.OPENROUTER_MODEL,
            temperature=0,
        )

    else:
        raise ValueError(f"Unknown PROVIDER: {config.PROVIDER}")


def get_embeddings():
    """
    Trả về Embeddings object dựa trên PROVIDER trong config.
    """
    if config.PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            api_key=config.OPENAI_API_KEY,
            model=config.OPENAI_EMBEDDING_MODEL,
            base_url=config.OPENAI_BASE_URL if config.OPENAI_BASE_URL else None,
        )

    elif config.PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            api_key=config.GOOGLE_API_KEY,
            model=config.GEMINI_EMBEDDING_MODEL,
        )

    elif config.PROVIDER == "anthropic":
        # Anthropic không có Embeddings API → dùng OpenAI
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            api_key=config.OPENAI_API_KEY,
            model=config.OPENAI_EMBEDDING_MODEL,
        )

    elif config.PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            base_url=config.OLLAMA_BASE_URL,
            model=config.OLLAMA_EMBEDDING_MODEL,
        )

    elif config.PROVIDER == "openrouter":
        # OpenRouter dùng OpenAI Embeddings API
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            api_key=config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            model="text-embedding-3-small",
        )

    else:
        raise ValueError(f"Unknown PROVIDER: {config.PROVIDER}")


if __name__ == "__main__":
    # Test
    print(f"Provider: {config.PROVIDER}")
    llm = get_llm()
    embeddings = get_embeddings()
    print(f"LLM: {llm}")
    print(f"Embeddings: {embeddings}")
