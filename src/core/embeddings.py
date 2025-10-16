import os
from typing import Optional
from dotenv import load_dotenv
load_dotenv() 

def get_embeddings(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    dimensions: Optional[int] = None,
): 
    """
    根据环境变量/参数选择并返回 LangChain Embeddings 实例。
    环境变量：
      - EMBEDDINGS_PROVIDER: "openai" | "azure" | "hf"
      - EMBEDDINGS_MODEL: 服务商具体的嵌入模型名称，如 "text-embedding-3-small"
      - EMBEDDINGS_DIM: 可选，整数
    """
    provider = (provider or os.getenv("EMBEDDINGS_PROVIDER") or "openai").lower()
    print(f"Using embeddings provider: {provider}")  # Debug 输出
    dimensions = dimensions or os.getenv("EMBEDDINGS_DIM") or 1536

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        model = model or os.getenv("EMBEDDINGS_MODEL") or "text-embedding-3-small"
        return OpenAIEmbeddings(model=model)

    if provider == "azure":
        from langchain_openai import AzureOpenAIEmbeddings
        model = model or os.getenv("EMBEDDINGS_MODEL") or os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT") or "text-embedding-3-small"
        return AzureOpenAIEmbeddings(model=model)

    if provider in {"hf", "huggingface"}:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings()

    raise ValueError(f"不支持的 EMBEDDINGS_PROVIDER: {provider}")
