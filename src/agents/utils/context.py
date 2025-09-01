# assemble_context.py
from __future__ import annotations
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Literal

from langchain_core.messages import (
    BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
)
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

# ---- 可选：严格“最近 k 轮”切窗（turn-aware） ----
def last_k_turns(
    messages: Sequence[BaseMessage],
    k: int = 3,
    include_system: bool = True,
) -> List[BaseMessage]:
    sys_msgs = [m for m in messages if include_system and isinstance(m, SystemMessage)]
    non_sys = [m for m in messages if not isinstance(m, SystemMessage)]
    turns: List[List[BaseMessage]] = []
    buf: List[BaseMessage] = []
    for m in non_sys:
        if isinstance(m, HumanMessage):
            if buf:
                turns.append(buf)
            buf = [m]
        else:
            if buf:
                buf.append(m)
            else:
                buf = [m]  # 罕见：非 Human 开头
    if buf:
        turns.append(buf)
    kept = [m for turn in turns[-k:] for m in turn]
    return [*sys_msgs, *kept]


def _tokens(msgs: Sequence[BaseMessage]) -> int:
    # 近似 token 计数（足够做预算控制）
    return count_tokens_approximately(list(msgs))  # type: ignore


def _as_text(chunk: Any) -> str:
    # 支持传入 str 或带 page_content 的 Document
    if isinstance(chunk, str):
        return chunk
    pc = getattr(chunk, "page_content", None)
    return pc if isinstance(pc, str) else str(chunk)


def assemble_context(
    state: Mapping[str, Any],
    *,
    # —— 总预算（上下文）——
    model_context_budget: int = 4000,

    # —— 窗口策略（二选一）——
    window_strategy: Literal["token_budget", "turns"] = "token_budget",
    window_turns: int = 3,
    start_on: Literal["human", "ai", "tool"] = "human",
    end_on: Sequence[Literal["human", "ai", "tool"]] = ("human", "tool"),
    include_system: bool = True,

    # —— 摘要（可选）——
    summary_provider: Optional[Callable[[Mapping[str, Any]], Optional[str]]] = None,
    summary_role: Literal["system", "assistant"] = "system",
    summary_soft_limit_tokens: int = 300,

    # —— RAG 注入（可选）——
    rag_retriever: Optional[Callable[[str, int], Iterable[Any]]] = None,
    rag_k: int = 3,
    rag_formatter: Optional[Callable[[str], str]] = None,

    # —— 其他 —— 
    min_window_tokens: int = 800,   # 给“最近窗口”留的最小份额，避免被摘要/RAG 挤没
) -> List[BaseMessage]:
    """
    组装“本次要喂给 LLM 的上下文消息列表”，不修改 state：
    - 从 state["messages"] 取原始历史
    - 注入：running summary（若提供） + RAG（若提供）
    - 窗口：按 token 预算或按最近 k 轮裁剪
    - 超预算处理：优先保窗口，先丢 RAG，再截短摘要

    期望的 state：
      state["messages"]: Sequence[BaseMessage]
      state["running_summary"]: Optional[str]  # 如果 summary_provider 用得上
    """
    history: Sequence[BaseMessage] = list(state.get("messages", []))

    # ---------- 1) 准备“固定/钉住”的前置消息：摘要 + RAG ----------
    pinned: List[BaseMessage] = []

    # 1a) 摘要（如有）
    summary_text: Optional[str] = None
    if summary_provider:
        summary_text = summary_provider(state)
        if summary_text:
            if summary_role == "assistant":
                pinned.append(AIMessage(content=summary_text))
            else:
                pinned.append(SystemMessage(content=summary_text))

    # 1b) RAG（如有）
    if rag_retriever:
        # 用“最近的人类提问”作为检索 query（先在 history 里找）
        last_user = next((m for m in reversed(history) if isinstance(m, HumanMessage)), None)
        query = last_user.content if last_user else ""
        if query:
            fmt = rag_formatter or (lambda s: f"[memory] {s.strip()}")
            for chunk in rag_retriever(query, rag_k) or []:
                text = _as_text(chunk).strip()
                if text:
                    pinned.append(SystemMessage(content=fmt(text)))

    # ---------- 2) 为窗口预留预算，并按策略构造“最近窗口” ----------
    pinned_budget = _tokens(pinned) if pinned else 0
    # 给窗口至少留一部分预算
    window_budget = max(model_context_budget - pinned_budget, min_window_tokens)

    if window_strategy == "turns":
        window = last_k_turns(history, k=window_turns, include_system=include_system)
        # 若仍超预算，逐步减少 k
        k = window_turns
        while _tokens([*pinned, *window]) > model_context_budget and k > 1:
            k -= 1
            window = last_k_turns(history, k=k, include_system=include_system)
    else:
        # token 预算切窗（推荐默认）
        # 先用“全预算”粗裁一遍，后面再用 pinned 调整
        window = trim_messages(
            history,
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=window_budget,
            start_on=start_on,
            end_on=tuple(end_on),
            include_system=include_system,
        )

    ctx: List[BaseMessage] = [*pinned, *window]

    # ---------- 3) 若仍超总预算：先丢 RAG，再截短摘要 ----------
    def drop_last_rag(msgs: List[BaseMessage]) -> bool:
        for i in range(len(msgs) - 1, -1, -1):
            m = msgs[i]
            if isinstance(m, SystemMessage) and m.content.strip().startswith("[memory]"):
                msgs.pop(i)
                return True
        return False

    def shorten_summary(msgs: List[BaseMessage]) -> bool:
        for i, m in enumerate(msgs):
            if isinstance(m, (SystemMessage, AIMessage)) and m.content == (summary_text or ""):
                # 粗略截短：按 token 目标对半压；你也可替换为“调用小模型生成更短摘要”
                text = m.content
                if not text:
                    return False
                # 简单字数近似：token≈char/3
                target_chars = max(50, summary_soft_limit_tokens * 3 // 2)
                if len(text) <= target_chars:
                    return False
                msgs[i] = type(m)(content=text[:target_chars] + " …")
                return True
        return False

    # 尽量把上下文压进预算里
    safe_guard = 0
    while _tokens(ctx) > model_context_budget and safe_guard < 10:
        safe_guard += 1
        # 先保住“窗口”，优先丢 RAG
        if drop_last_rag(pinned):
            ctx = [*pinned, *window]
            continue
        # 再截短摘要
        if shorten_summary(pinned):
            ctx = [*pinned, *window]
            continue
        # 实在不行，进一步收缩窗口（只在 token 策略下生效）
        if window_strategy == "token_budget" and window_budget > 200:
            window_budget = max(200, int(window_budget * 0.8))
            window = trim_messages(
                history,
                strategy="last",
                token_counter=count_tokens_approximately,
                max_tokens=window_budget,
                start_on=start_on,
                end_on=tuple(end_on),
                include_system=include_system,
            )
            ctx = [*pinned, *window]
            continue
        break

    return ctx
