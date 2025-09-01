from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from typing import List
import os

# ===== Public API ======
# 创建或更新向量数据库
def create_vector_db(
    jsonl_path: str,
    persist_directory: str = "chroma_db",
    collection_name: str = "chat_history",
    metadata_func=None,
    block_len: int = 6,
    overlap: int = 3,
    sep: str = "\n",
    keep_tail: bool = True,
    mode: str = "upsert",
    embedding: HuggingFaceEmbeddings = HuggingFaceEmbeddings()
):
    docs = load_docs_from_jsonl(
        jsonl_path=jsonl_path,
        block_len=block_len,
        overlap=overlap,
        sep=sep,
        keep_tail=keep_tail,
        metadata_func=metadata_func,
    )

    vectorstore = upsert_docs_to_chroma(
        docs=docs,
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding=embedding,
        mode=mode
    )

    return vectorstore

# 获取检索工具 在agent构建时调用，绑定到llm
def get_retriever_tool(persist_directory, collection_name, embedding=HuggingFaceEmbeddings()):
    vectorstore = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # K is the amount of chunks to return
    )
    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="chat_history_retriever",
        description="A tool for retrieving chat history"
    )
    return retriever_tool



# ====== Internal API ======
# 从 JSONL 文件加载文档并分块
def load_docs_from_jsonl(
    jsonl_path: str,
    content_key: str = "text",
    jq_schema: str = ".",
    json_lines: bool = True,
    metadata_func=None,
    block_len: int = 6,
    overlap: int = 3,
    sep: str = "\n",
    keep_tail: bool = True,
) -> List[Document]:
    loader = JSONLoader(
        file_path=jsonl_path,
        jq_schema=jq_schema,
        content_key=content_key,
        json_lines=json_lines,
        metadata_func=metadata_func
    )
    raw_docs = loader.load()
    # 进行分块
    chunked_docs = window_lines(
        docs=raw_docs,
        block_len=block_len,
        overlap=overlap,
        sep=sep,
        keep_tail=keep_tail
    )
    return chunked_docs

# 将文档插入 Chroma 向量数据库
def upsert_docs_to_chroma(
    docs,
    persist_directory: str = "chroma_db",
    collection_name: str = "chat_history",
    embedding=HuggingFaceEmbeddings(),
    mode="upsert"
    ):
    # 如果路径不存在，则创建
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    # 向量数据库存储信息
    try:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        return vectorstore
    except Exception as e:
        print(f"Error occurred while creating vectorstore: {e}")
        return None

    


#====== Private helpers ======

def keep_metadata(record, metadata):
    metadata["role"] = record["role"]
    metadata["timestamp"] = record["timestamp"]
    return metadata



# 窗口分段组合
def window_lines(
    docs: List[Document],
    block_len: int = 6,
    overlap: int = 3,
    sep: str = "\n",
    keep_tail: bool = True,
) -> List[Document]:
    """
    将原始逐行文档划分为重叠窗口，仅写入标量类型的 metadata，
    以避免 Chroma 对复杂元数据类型（list/dict/None）报错。
    """
    # ---- 参数校验 ----
    if block_len <= 0:
        raise ValueError("block_len must be > 0")
    if overlap < 0 or overlap >= block_len:
        raise ValueError("overlap must satisfy 0 <= overlap < block_len")

    n = len(docs)
    if n == 0:
        return []

    out: List[Document] = []
    stride = block_len - overlap
    i = 0

    def _first_non_none(seq):
        for x in seq:
            if x is not None:
                return x
        return None

    def _pack(block: List[Document], start_idx: int, is_tail: bool = False) -> Document:
        end_idx = start_idx + len(block) - 1

        # 提取首/尾标量元数据（如果存在）
        roles = []
        timestamps = []
        for d in block:
            md = d.metadata if isinstance(d.metadata, dict) else {}
            roles.append(md.get("role"))
            timestamps.append(md.get("timestamp"))

        start_role = _first_non_none(roles)
        end_role = _first_non_none(reversed(roles))
        start_ts = _first_non_none(timestamps)
        end_ts = _first_non_none(reversed(timestamps))

        metadata = {
            "window_index": len(out),
            "line_start": start_idx,
            "line_end": end_idx,
            "block_len": len(block),
            "overlap": overlap,
            "stride": stride,
            "tail": is_tail,
        }

        # 仅在有值时写入（并确保为标量）
        if start_role is not None:
            metadata["start_role"] = str(start_role)
        if end_role is not None:
            metadata["end_role"] = str(end_role)
        if start_ts is not None:
            metadata["start_ts"] = start_ts if isinstance(start_ts, (int, float, bool)) else str(start_ts)
        if end_ts is not None:
            metadata["end_ts"] = end_ts if isinstance(end_ts, (int, float, bool)) else str(end_ts)

        return Document(
            page_content=sep.join((d.page_content or "") for d in block),
            metadata=metadata,
        )

    # ---- 常规完整窗口 ----
    while i + block_len <= n:
        block = docs[i:i + block_len]
        out.append(_pack(block, start_idx=i, is_tail=False))
        i += stride

    # ---- 尾窗（仅当有“余数”且用户需要时）----
    if keep_tail:
        rem = n - i
        if 0 < rem < block_len:
            start_tail = max(n - block_len, 0)  # 尽量向右对齐
            if out and start_tail == out[-1].metadata["line_start"]:
                start_tail = min(start_tail + stride, max(n - 1, 0))
                start_tail = min(start_tail, max(n - block_len, 0))

            block = docs[start_tail:n]
            out.append(_pack(block, start_idx=start_tail, is_tail=True))

    return out


if __name__ == "__main__":
    vs = create_vector_db(
        jsonl_path="chat_session.jsonl",
        persist_directory="chroma_db",
        collection_name="chat_history"
    )
    if isinstance(vs, Chroma):
        print("Vector DB created/updated successfully.")
