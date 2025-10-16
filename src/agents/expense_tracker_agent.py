from typing import Any, Literal, Optional, TypedDict
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langmem import create_manage_memory_tool, create_search_memory_tool
from core import get_model, settings
from agents.utils.context import assemble_context
from agents.utils.chat_log import append_msg
from agents.utils.rag_tool import get_retriever_tool
from agents.utils.store import get_store
from agents.utils.db_repo import get_db
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from pydantic import BaseModel, Field, ConfigDict, confloat
import datetime as dt
import json
from dotenv import load_dotenv
load_dotenv()  # 读取 .env


# 统一数据库适配器实例（按环境 DB_DIALECT 切换）
DB = get_db()
DB.init()

# 持久化键值对存储（按环境 STORE_DIALECT 切换）
store = get_store()



# 数据库写入结果类型
class DBResult(TypedDict, total=False):
    status: Literal["inserted", "error", "skipped"]
    rowid: int
    error: str

# 状态
class AgentState(MessagesState):
    intent: str | None  # 意图
    parsed: dict | None  # 解析结果
    awaiting: Optional[Literal["fill"]]  # 等待填充状态
    pending_fields: Optional[list[str]]  # 待填充字段
    draft: Optional[dict]  # 信息不完整的草稿
    validated: Optional[bool]  # 是否验证通过
    db_result: Optional[DBResult]  # 数据库写入结果
    query_plan: Optional[dict]  # 查询计划
    query_result: Optional[dict]  # 查询结果


# 工具
# rag_retriever = get_retriever_tool(persist_directory="chroma_db", collection_name="chat_history")  # RAG 检索工具（测试时暂不启用）

# LangMem
_NAMESPACE = ("memories",)
manage_mem_tool = create_manage_memory_tool(namespace=_NAMESPACE)
search_mem_tool = create_search_memory_tool(namespace=_NAMESPACE)

# 仅保留 LangMem 工具，便于测试自然调用 _search_mem_tool
tools = [manage_mem_tool, search_mem_tool]
tool_node = ToolNode(tools)


# 路由入口的占位节点，不改状态 | 写入用户原始消息
def entry_node(state: AgentState, config: RunnableConfig) -> AgentState:
    user_text = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)).content
    append_msg("user", user_text)
    return {}

# 路由判断，是否有信息待补充
def route_from_entry(state: AgentState) -> str:
    return "handle_fill" if state.get("awaiting") == "fill" else "classify"

# 是否补充完成
def route_after_fill(state: AgentState) -> str:
    # 如果已不再等待且准备好了 parsed，就进入 validate；否则先去 respond
    return "validate" if state.get("parsed") else "respond"

# 意图分类
classify_instructions = """你是记账助手，需要为用户的输入判定意图标签：
- log_expense：描述新的收支记录（“买了/花了/收入/转账”）。
- query_summary：询问历史收支统计或明细（“过去一周早餐花了多少”“最近三次买咖啡”）。
- related_chat：与消费有关的闲聊、追问、复述或寻求建议，但无新统计需求。消费可能是隐含的
- other：完全无关的内容。
请只输出提供的标签之一，不要新造标签。"""

# 意图分类模型
class IntentOut(BaseModel):
    """记账意图分类输出"""
    intent: Literal["log_expense", "query_summary", "related_chat", "other"] = Field(
        ...,
        description="""
        用户意图分类：
        - log_expense：新的可入账收支描述。
        - query_summary：询问历史收支统计或需要数据库检索的汇总问题。
        - related_chat：围绕既有记录或偏好闲聊，未请求数据库查询。
        - other：其余与记账无关或含义不明确的内容。
        """
    )

# 意图分类节点
def classify_intent(state: AgentState, config: RunnableConfig) -> AgentState:
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    llm_struct = llm.with_structured_output(IntentOut)
    user_text = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)).content
    result: IntentOut = llm_struct.invoke([SystemMessage(content=classify_instructions),
                                           *assemble_context(state=state, window_strategy="turns", window_turns=6, include_system=False),
                                           HumanMessage(content=user_text)])
    # 可选：把结果也记录到AI消息里（便于可观测）
    ai = AIMessage(content=f"intent={result.intent}")
    return {
        "messages": [ai],
        "intent": result.intent,
    }

# 条件函数，判断是否为可记账的消费/收入类表述
def is_log_expense(state: AgentState) -> bool:
    return state.get("intent") == "log_expense"


# —— Related Chat 专用系统提示 ——
RELATED_CHAT_SYS = """
你是记账助手。用户正在询问与历史有关的问题（如过去的花费、已记录的信息、偏好等）。
请先调用已注入的历史检索类工具获取相关片段，再基于检索结果用一条简洁自然的中文回答；如合适，附带1条关怀式追问。
要求：
- 先检索再作答；可以调用一个或多个检索工具（按工具自身说明使用）。
- 当用户问题包含引号内容时，优先把引号内词作为检索关键词。
- 输出仅一条消息，不要暴露内部术语或工具细节。
"""


def route_after_classify(state: AgentState) -> str:
    intent = state.get("intent")
    if intent == "log_expense":
        return "extract"
    if intent == "query_summary":
        return "plan_query"
    if intent == "related_chat":
        return "respond_related"
    return "respond"


def respond_related(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    专门处理 related_chat：让 LLM 自主调用检索工具并组织回答。
    """
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    llm = llm.bind_tools(tools)

    msgs: list[BaseMessage] = [
        SystemMessage(content=RELATED_CHAT_SYS),
        *assemble_context(state=state, window_strategy="turns", window_turns=6, include_system=False),
    ]

    ai = llm.invoke(msgs)
    return {"messages": [ai]}

# 信息提取模型与提示词
class ExtractOut(BaseModel):
    # 最小两槽（允许缺失；缺了就走等待补充）
    item: Optional[str] = Field(default=None, description="商品/消费内容的简短名称，如 '早餐'、'哑铃'")
    amount: Optional[float] = Field(default=None, gt=0, description="金额，单位元，>0")

    # 其余可选（以后想用再用）
    currency: str = Field(default="CNY")
    occurred_at_text: Optional[str] = Field(default=None)
    occurred_at_iso: Optional[str] = Field(default=None)
    category: Optional[str] = Field(default=None)
    merchant: Optional[str] = Field(default=None)
    note: Optional[str] = Field(default=None)

extract_instructions = """你是中文记账抽取助手。把用户的消费描述解析为结构化JSON。
遵循下面字段：
- item: 商品/消费内容的简短名称（尽量2~6个字，如“早餐”“哑铃”）。无法确定可留空。
- amount: 金额（单位元，浮点数，大于0）。无法确定可留空。
- currency: 默认 "CNY"
- occurred_at_text: 原始时间短语（如“今天早上”“刚刚”），没有可留空
- occurred_at_iso: 若能确定具体时间，用 ISO8601 到分钟；否则留空
- category, merchant, note: 可留空

只输出 JSON，不要多余文本。"""


# 查询意图解析模型与提示词
class QueryExpenseOut(BaseModel):
    """查询类问题的结构化计划"""

    metric: Literal["sum", "avg", "count", "list", "latest"] = Field(
        ...,
        description="聚合目标，sum=求总金额，avg=平均金额，count=条目数，list=返回明细，latest=最近一条记录。",
    )
    time_scope: str | None = Field(
        default=None,
        description="用户问题中的原始时间短语，如‘过去一周’。",
    )
    start_iso: str | None = Field(
        default=None,
        description="时间范围起点（含），ISO8601 日期，例 2024-03-11。",
    )
    end_iso: str | None = Field(
        default=None,
        description="时间范围终点（含），ISO8601 日期。",
    )
    item_keywords: list[str] = Field(
        default_factory=list,
        description="与消费条目相关的关键词列表（小写，便于 LIKE 查询）。",
    )
    categories: list[str] = Field(
        default_factory=list,
        description="用户提到的品类标签，如‘食品’。",
    )
    merchants: list[str] = Field(
        default_factory=list,
        description="涉及的商家名称。",
    )
    notes: str | None = Field(
        default=None,
        description="其他可用于过滤的描述，如支付方式。",
    )


query_plan_instructions = """你是中文记账查询规划助手，需要把用户关于历史收支的提问解析成 JSON。
字段要求：
- metric: 选择 {"sum","avg","count","list","latest"} 中最契合的问题需求；当用户询问“上一次/最近一次”时使用 latest。
- time_scope: 填写用户的时间短语原文（如“过去一周”）；若未提到填 null。
- start_iso / end_iso: 推理得到的闭区间起止日期（ISO8601，精确到日，缺失填 null；若只给具体日期就同一天）。
- item_keywords: 归纳与消费条目相关的关键词列表（用原词或其小写形式）。
- categories: 若提到如“食品/固定资费/交通”等品类词，放入该数组；否则为空数组。
- merchants: 出现具体商家时列出；否则为空数组。
- notes: 其他能帮助数据库过滤的描述（如“微信”“公司报销”）；没有填 null。
约束：
1. 不要做金额运算或臆造字段。
2. 未给出的字段填 null 或空列表。
3. 输出必须是 JSON，禁止额外文本。
"""


def plan_query(state: AgentState, config: RunnableConfig) -> AgentState:
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    llm_struct = llm.with_structured_output(QueryExpenseOut)

    user_text = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)).content

    result: QueryExpenseOut = llm_struct.invoke([
        SystemMessage(content=query_plan_instructions),
        SystemMessage(content=f"current datetime: {dt.datetime.now().isoformat(timespec='minutes')}"),
        *assemble_context(state=state, window_strategy="turns", window_turns=6, include_system=False),
        HumanMessage(content=user_text),
    ])

    audit = AIMessage(
        content=f"[plan_query] metric={result.metric}, start={result.start_iso}, end={result.end_iso}, "
        f"keywords={result.item_keywords}, "
        f"categories={result.categories}, merchants={result.merchants}, notes={result.notes}"
    )

    plan_dict = result.model_dump()
    start_iso = plan_dict.get("start_iso")
    end_iso = plan_dict.get("end_iso")
    if start_iso and end_iso and end_iso < start_iso:
        plan_dict["start_iso"], plan_dict["end_iso"] = end_iso, start_iso

    return {
        "messages": [audit],
        "query_plan": plan_dict,
        "query_result": None,
    }


def _iso_to_next_day(iso_date: str) -> str:
    date_obj = dt.date.fromisoformat(iso_date)
    return (date_obj + dt.timedelta(days=1)).isoformat()


def run_query(state: AgentState, config: RunnableConfig) -> AgentState:
    plan_state = state.get("query_plan") or {}
    plan = dict(plan_state)
    if not plan:
        return {
            "messages": [AIMessage(content="[run_query] skipped: empty plan")],
            "query_result": {"status": "error", "message": "查询计划为空"},
        }

    start_iso = plan.get("start_iso")
    end_iso = plan.get("end_iso")
    if start_iso and end_iso and end_iso < start_iso:
        start_iso, end_iso = end_iso, start_iso
    if start_iso:
        plan["start_iso"] = start_iso
    if end_iso:
        plan["end_iso"] = end_iso
        plan.setdefault("_end_exclusive", _iso_to_next_day(end_iso))

    try:
        result = DB.summarize_transactions(plan)
        audit_msg = f"[run_query] rows={result.get('total_rows', 0)} metric={result.get('metric')}"
        return {
            "messages": [AIMessage(content=audit_msg)],
            "query_result": result,
        }
    except Exception as exc:
        return {
            "messages": [AIMessage(content=f"[run_query_error] {exc}")],
            "query_result": {"status": "error", "message": str(exc)},
        }


# 信息提取节点
def extract_struct(state: AgentState, config: RunnableConfig) -> AgentState:
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    llm_struct = llm.with_structured_output(ExtractOut)

    # 取最后一条用户输入
    user_text = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)).content

    result: ExtractOut = llm_struct.invoke([
        SystemMessage(content=extract_instructions),
        HumanMessage(content=user_text),
    ])

    # 审计信息（便于观察抽取是否合理）
    audit = AIMessage(
        content=f"[extract] item={result.item!r}, amount={result.amount}, "
                f"t='{result.occurred_at_text}', iso='{result.occurred_at_iso}', "
                f"cat='{result.category}', merchant='{result.merchant}'"
    )

    # 注意：这里只“承接原样”，不做校验与归一化（下一步在 validate_normalize 做）
    return {
        "messages": [audit],
        "parsed": result.model_dump(),
    }

# 标准化辅助函数
# 时间归一化
def _normalize_time(text: Optional[str], iso: Optional[str]) -> str:
    """
    输出 ISO8601（到分钟）。优先用 occurred_at_iso；否则根据中文相对时间粗略推断；最后兜底当前时间。
    """
    now = dt.datetime.now()
    if iso:
        try:
            t = dt.datetime.fromisoformat(iso.replace("Z",""))
            return t.replace(second=0, microsecond=0).isoformat(timespec="minutes")
        except Exception:
            pass

    t = now
    if text:
        if "昨天" in text: t = now - dt.timedelta(days=1)
        if "早" in text: t = t.replace(hour=8, minute=0, second=0, microsecond=0)
        elif "中午" in text: t = t.replace(hour=12, minute=0, second=0, microsecond=0)
        elif "晚" in text: t = t.replace(hour=19, minute=0, second=0, microsecond=0)
        elif "刚刚" in text or "现在" in text or "今天" in text:
            t = t.replace(second=0, microsecond=0)
    return t.isoformat(timespec="minutes")

# 金额换算
def _yuan_to_cents(yuan: float) -> int:
    return int(round(float(yuan) * 100))

# 校验和标准化节点
def validate_normalize(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    目标：
    - 必填槽：item、amount
    - 若缺：进入 awaiting="fill"，把当前 parsed 挪到 draft，并问缺什么
    - 若齐：归一化并标记 validated=True（下一步就能直接写库）
    """
    raw = state.get("parsed") or {}
    item = (raw.get("item") or "").strip() or None
    amount = raw.get("amount")  # 元（float）
    missing = []

    if not item:
        missing.append("item")
    if amount is None or float(amount) <= 0:
        missing.append("amount")

    # 统一归一化（即便缺少，也先把能定的东西写入 draft）
    occurred_at = _normalize_time(raw.get("occurred_at_text"), raw.get("occurred_at_iso"))
    currency = raw.get("currency") or "CNY"
    category = raw.get("category") or None
    merchant = raw.get("merchant") or None
    note = raw.get("note") or None
    source_msg = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)).content

    # 先组一份“草稿/或规范化载体”
    proto = {
        "type": "expense",
        "item": item,
        "currency": currency,
        "occurred_at": occurred_at,
        "category": category,
        "merchant": merchant,
        "note": note,
        "source_message": source_msg,
    }
    if amount is not None and float(amount) > 0:
        proto["amount_yuan"] = float(amount)

    # 分支：缺槽 -> 等补充
    if missing:
        return {
            "messages": [AIMessage(content=f"[need_fill] missing={missing}")],  # 审计信息可留可去
            "awaiting": "fill",
            "pending_fields": missing,
            "draft": proto,  # 把工作载体交给 draft，等待下一轮补充
            "validated": False,
            "parsed": None,   # 暂时留空
        }

    # 分支：齐全 -> 规范化为可落库 payload
    normalized = dict(proto)
    normalized["amount_cents"] = _yuan_to_cents(proto.pop("amount_yuan"))
    # 也可以在这里做更多：比如 item 简化清洗、category 映射等

    audit = AIMessage(content=f"[validate] ok: {normalized['item']} ¥{normalized['amount_cents']/100:.2f} @ {normalized['occurred_at']}")
    return {
        "messages": [audit],
        "validated": True,
        "parsed": normalized,  # ⬅️ 现在 parsed 就是“可写库 payload”
        "awaiting": None,
        "pending_fields": None,
    }

# 条件函数，检查validate的状态
def is_validated(state: AgentState) -> bool:
    return state.get("validated", False)

def write_db(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    使用仓库 DB 将已规范化 payload 写入交易表。
    """
    if not state.get("validated") or not state.get("parsed"):
        return {"messages": [AIMessage(content="[db] skipped: not validated")]}

    p = dict(state["parsed"])  # 复制一份，避免副作用

    try:
        txn_id = DB.insert_transaction(p)

        return {
            "messages": [AIMessage(content=f"[db] inserted id={txn_id}")],
            "db_result": {"status": "inserted", "rowid": txn_id},
        }
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"[db_error] {e}")],
            "db_result": {"status": "error", "error": str(e)},
        }






# —— Respond 辅助：提示词 & 摘要 —— 
FINALIZE_SYS = """
你是记账助手的“最终回复生成器（finalizer）”。你会看到：
- 完整的消息历史（包含审计信息，如 [extract] / [validate] / [db] / [handle_fill]）。
- 一个工具消息 state_snapshot，提供当前状态与结构化字段。

任务：基于以上信息，生成一条**给用户看的**自然中文回复（单条消息）。

场景与要求（择一触发，按此优先级）：
0) 若 state_snapshot.intent == "query_summary":
   - 必须引用 state_snapshot.query_result 中的字段；禁止自行编造金额或条数。
   - 若 query_result.status == "error"：简要说明无法查询的原因，并引导用户稍后再试或调整问题。
   - 若 query_result.total_rows == 0：告诉用户未找到符合条件的记录，鼓励继续记账或尝试其它条件。
   - 若存在金额字段：使用 query_result.total_amount_yuan（保留两位小数，前缀 ¥）。
   - 若 query_result.metric == "latest" 且存在 latest_record：说明最近一笔的金额（用 latest_amount_yuan，保留两位小数，前缀 ¥）与发生时间；如记录含 item/category 可自然引用。
   - 如包含时间范围或关键词，可自然引用 query_plan/time_scope，帮助用户理解统计范围。
   - 保持语气专业友好，可附一句关怀或建议。
1) 若 state_snapshot.db_result.status == "inserted":
   - 生成确认语：包含 item、金额（用 ¥，两位小数）、时间（occurred_at 到分钟）。
   - 如有 merchant/category 可适度附加；语气简洁自然；不要出现方括号审计字样或字段名。
   - 然后再附上一句关切的话。
2) 若 state_snapshot.db_result.status == "error":
   - 简短致歉 + 概述错误要点（提炼，不要技术堆栈），给出一条建议（如“稍后再试/稍作修改重发”）。
3) 若 state_snapshot.awaiting == "fill":
   - 只就 state_snapshot.pending_fields 追问（单句，15~40字，举1个示例，口语自然；不要列表/多段）。
   - 语气可爱一点
4) 若 state_snapshot.intent == "related_chat":
   - 必须查询历史消息！！！
   - 必须构造 tool_call
   - 必须调用 chat_history_retriever 在历史记录中查询；把用户问题里的关键词作为 query（若出现引号内词，优先用引号内的词）。
   - 回复用户的问题，展示关怀，然后适当抛出新的疑问（比如“那今天呢”，”那个哑铃的使用情况如何？”）
5) 若 state_snapshot.intent == "other":
   - 适当简单回复用户的问题，但牢记自己是记账智能体的身份。
   - 给一句简短引导并给出两种可模仿的记账示例（同一行用分隔符；避免多段）。
6) 其余情况：
   - 若 parsed 基本齐全但尚未写库：给出简洁确认或引导。
   - 否则给出简短提示，说明如何继续。

风格：
- 口语化、清爽、无前缀、无列表、无内部术语；不要暴露审计标签或内部字段名。
"""




def respond(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    统一出口：交给 LLM（finalizer）根据完整历史+状态快照生成“用户可见”的最终一句。
    """
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    llm = llm.bind_tools(tools)

    # 准备状态快照（给 LLM 做决策，不直接面向用户）
    snapshot = {
        "intent": state.get("intent"),
        "awaiting": state.get("awaiting"),
        "pending_fields": state.get("pending_fields"),
        "validated": state.get("validated"),
        "db_result": state.get("db_result"),
        "draft": state.get("draft"),
        "parsed": state.get("parsed"),
        "query_plan": state.get("query_plan"),
        "query_result": state.get("query_result"),
    }

    tool_call_id = "final-ctx-0001"
    msgs: list[BaseMessage] = [
        SystemMessage(content=FINALIZE_SYS),
        # 取最近若干轮历史，含审计信息；你已有这个工具，直接复用
        *assemble_context(state=state, window_strategy="turns", window_turns=6, include_system=False),
        AIMessage(content="", tool_calls=[{
            "id": tool_call_id,
            "type": "tool_call",
            "name": "state_snapshot",
            "args": {}
        }]),
        ToolMessage(
            name="state_snapshot",
            tool_call_id=tool_call_id,
            content=json.dumps(snapshot, ensure_ascii=False),
        ),
        HumanMessage(content="请基于以上历史与状态，生成给用户看的最终一句回复。")
    ]

    try:
        msg = llm.invoke(msgs)
        if msg.additional_kwargs.get("tool_calls"): return {"messages": [msg]}
        out = {
            "messages": [AIMessage(content=msg.content, additional_kwargs={"visibility": "user"})]
        }
        append_msg("assistant", msg.content)
    except Exception:
        # 兜底：极少数情况下 LLM 异常，用一个通用模板
        out = {"messages": [AIMessage(
            content="这条信息我已处理；如果需要补充，请直接告诉我金额或商品名称即可～",
            additional_kwargs={"visibility": "user"}
        )]}

    # 清一次 db_result，避免下一轮重复确认
    db = state.get("db_result") or {}
    if db.get("status") in ("inserted", "error"):
        out["db_result"] = {}

    if state.get("query_result"):
        out["query_result"] = None
        out["query_plan"] = None

    return out

# handle_fill的system prompt
HANDLE_FILL_SYS = """
你是 handle_fill 的“补充判断器”。当 awaiting="fill" 时被调用。
你的任务：基于“最近几轮对话窗口”和工具消息 context_bundle（含 draft、pending），
**重新抽取一份尽可能完整的新 slots（ExtractOut）**，而不是机械继承 draft 旧值。
并根据语义选择 action ∈ {"fill","new_log","cancel","cancel_then_new","unrelated"}。

【可见上下文】
1) 最近对话窗口（系统已附带）：包含本轮用户话和前几轮关键信息。
2) context_bundle（工具消息）：
   - draft：上一轮校验后形成的“草稿载体”，仅供参考（含：item、amount_yuan、occurred_at、currency、category、merchant、note 等）。
   - pending：仍待补的字段列表（如 ["amount"] 或 ["item"]）。

【核心要求（非常重要）】
- 你要**从零开始**对“最近几轮对话 + 本轮话”做整体理解与抽取，生成 **新的、尽可能完整** 的 slots。
- 可以把 draft 作为线索，但**不要**无脑拷贝旧值；若历史上下文已能确定字段（如 item/时间），应写入到新 slots 中。
- 例如：上一轮用户说“我早上买了早餐”，本轮只说“10元”，则新 slots 应包含：item="早餐"，amount=10.0，occurred_at_*（若能从上下文判断），currency="CNY" 等。

【动作定义与判定优先级】
1) cancel_then_new：用户明确“先取消上笔，再记一笔新的”（如“那条算了，重新记：星巴克28”）。
2) cancel：仅取消旧草稿（“别记了/上一条作废/取消”），且没有紧随其后的新账描述。
3) new_log：用户**另起一笔**，与草稿不是同一条（常见触发词：“另外/再记/还有/第二笔/顺便”或内容与草稿明显不同且较完整）。
4) fill：仍是在补**同一条**草稿；本轮信息与历史合起来，凑出更完整的新 slots（哪怕本轮只说了数字，也要把历史里的 item/时间补齐到新 slots）。
5) unrelated：与记账无关或语义极度不明确。

【抽取规范（用于构造 slots: ExtractOut）】
- item：尽量2~6字（如“早餐/咖啡/哑铃”）。若能从上下文确定，应写入。
- amount：单位元，float，>0。口语规则：
  * “10/10.5/10元/10块/十/三十五/3块5/两块三/5毛/1块2毛”等要正确转数值：
    - “3块5”→3.5；“两块三”≈2.3；“5毛”→0.5；“1块2毛”→1.2；中文数字要转阿拉伯数。
  * 不确定不要臆造。
- currency：默认 "CNY"，除非对话里明确其它币种。
- 时间：
  * occurred_at_text：保留相对时间短语（“刚刚/今天/中午/昨天晚上/上周五” 等）。
  * occurred_at_iso：仅当能确定到“年月日+时分”时填写 ISO8601（到分钟，如“2025-08-26T08:00”）。
- category/merchant/note：对话中若出现则填写，否则留空。

【同一条 vs 新的一条 的判断要点】
- 若没有“另外/再记/第二笔”等串联词，且语义在延续补充（尤其 pending 所缺字段被补），一般判为 fill。
- 若出现明显串联词，或当前句给出了与草稿**不同**的 item/商户/时间且结构较完整，判为 new_log。
- 明确“先取消旧的再记”→ cancel_then_new。

【输出（DecisionOut）】
- action：按以上规则判定。
- confidence：0~1，正常给 0.6~0.9；特别不确定可降到 <0.4。
- slots：本次“重抽”的**完整新结果**（未提及的字段可留空 None；但能从上下文确定的请填上）。
- reason：简短中文解释（如“补金额且上下文确定 item=早餐 → fill；抽取 slots 已补齐”）。

请严格按以上要求决策并输出结构化结果。不要输出额外话语。
"""

# handle_fill的llm调用结果
class DecisionOut(BaseModel):
    action: Literal[
        "fill",  # 本句话是之前draft的补充或修正，并且现在可能是完整合法的 --> 把slots放到parsed并清空draft和pending --> 后续进validate
        "new_log",  # 本句话与之前的draft无关，是一条新的记录 --> 把slots放到parsed并保留draft和pending --> 后续进validate
        "cancel",  # 用户明确表示取消之前的draft和pending_field相关内容  --> 清空draft和pending --> 后续进respond
        "cancel_then_new",  # 用户取消之前的draft后，开始一个新的记录 --> 把slots放到parsed并清空draft和pending --> 后续进validate
        "unrelated",  # 用户的表述与之前的draft无关，也不想记账 --> 后续进respond
    ]
    confidence: float = Field(0.5, ge=0, le=1)  # 置信度，暂时忽略
    slots: ExtractOut = Field(default_factory=ExtractOut)
    reason: str = ""  # 原因审计


# 处理补充信息
def handle_fill(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    await='fill' 时进入：
    
    """

    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    llm_struct = llm.with_structured_output(DecisionOut)

    draft = state.get("draft") or {}
    pending = state.get("pending_fields") or []
    draft_and_pending_reference = {
        "draft": draft,
        "pending": pending
    }

    tool_call_id = "ctx-0001"
    msg = [
        SystemMessage(content=HANDLE_FILL_SYS),
        *assemble_context(state=state, window_strategy="turns", window_turns=3, include_system=False),
        AIMessage(content="", tool_calls=[{          # 声明一次“工具调用”
            "id": tool_call_id,
            "type": "tool_call",
            "name": "context_bundle",
            "args": {}
        }]),
        ToolMessage(                                  # 真正把字典塞给模型
            name="context_bundle",
            tool_call_id=tool_call_id,
            content=json.dumps(draft_and_pending_reference, ensure_ascii=False)
            # artifact=my_dict  # 可选：完整大对象放 artifact，不会发给模型
        ),
        HumanMessage(content="请基于以上上下文，输出结构化结果")

    ]

    result: DecisionOut = llm_struct.invoke(msg)

    # 创建决策审计消息，respond 节点可以通过 assemble_context 读取到这个信息
    decision_msg = AIMessage(
        content=f"[handle_fill] 决策：{result.action} | 原因：{result.reason}"
    )

    # CONF = 0.72  # 经验阈值，低于则先澄清（暂未启用）
    parsed = result.slots.model_dump(exclude_none=True)
    if result.action in ["fill", "cancel_then_new"]:
        return {
            "messages": [decision_msg],
            "parsed": parsed,
            "draft": {},
            "pending_fields": None,
            "awaiting": None,
        }
    elif result.action == "new_log":
        return {
            "messages": [decision_msg],
            "parsed": parsed,
        }
    elif result.action == "cancel":
        return {
            "messages": [decision_msg],
            "draft": {},
            "pending_fields": None,
            "awaiting": None,
        }
    else:
        return {
            "messages": [decision_msg],
        }
    



# 构建图
graph = StateGraph(AgentState)

# 节点注册
graph.add_node("entry", entry_node)
graph.add_node("classify", classify_intent)
graph.add_node("plan_query", plan_query)
graph.add_node("extract", extract_struct)
graph.add_node("validate", validate_normalize)
graph.add_node("write_db", write_db)
graph.add_node("run_query", run_query)
graph.add_node("handle_fill", handle_fill)
graph.add_node("respond", respond)
graph.add_node("respond_related", respond_related)
graph.add_node("tools", tool_node)


# 边连接
graph.add_edge(START, "entry")
graph.add_conditional_edges(
    "entry",
    route_from_entry,
    {
        "handle_fill": "handle_fill",
        "classify": "classify"
    }
)
graph.add_conditional_edges(
    "classify",
    route_after_classify,
    {
        "extract": "extract",
        "plan_query": "plan_query",
        "respond_related": "respond_related",
        "respond": "respond",
    }
)
graph.add_edge("plan_query", "run_query")
graph.add_edge("run_query", "respond")
graph.add_edge("extract", "validate")
graph.add_conditional_edges(
    "validate",
    is_validated,
    {
        True: "write_db",
        False: "respond"
    }
)
graph.add_edge("write_db", "respond")
graph.add_conditional_edges(
    "handle_fill",
    route_after_fill,
    {"validate": "validate", "respond": "respond"},
)
graph.add_conditional_edges(
    "respond",
    tools_condition,
    {"tools": "tools", END: END}
)
graph.add_edge("tools", "respond")

# 相关聊天专用的工具回路，避免影响既有 respond 工具流
tool_node_related = ToolNode(tools)
graph.add_node("tools_related", tool_node_related)
graph.add_conditional_edges(
    "respond_related",
    tools_condition,
    {"tools": "tools_related", END: END},
)
graph.add_edge("tools_related", "respond_related")

app = graph.compile(checkpointer=MemorySaver(), store=store)


# ======== CLI & 快速测试 ========
import time

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

def _print_last_ai(state_out):
    from langchain_core.messages import AIMessage
    last_ai = next((m for m in reversed(state_out["messages"]) if isinstance(m, AIMessage)), None)
    print(last_ai.content if last_ai else "(未返回 AI 消息)")

if __name__ == "__main__":

    # 1) 可视化图结构
    # from draw_graph import draw_graph
    # draw_graph(app)


    # 2) 快速双步测试（缺金额 → 补金额 → 入库）
    print(">>> 快速测试：缺金额 → 补金额")
    tid = f"demo-{int(time.time())}"
    out1 = app.invoke(
        {"messages": [HumanMessage(content="我早上买了早餐")]},
        config={"configurable": {"model": settings.DEFAULT_MODEL, "thread_id": tid}},
    )
    _print_last_ai(out1)

    out2 = app.invoke(
        {"messages": [HumanMessage(content="10元")]},
        config={"configurable": {"model": settings.DEFAULT_MODEL, "thread_id": tid}},
    )
    _print_last_ai(out2)

    # 3) 进入交互模式
    print("\n记账智能体已启动。直接输入开始；输入 /new 开新会话；输入 exit 退出。")
    thread_id = f"cli-{int(time.time())}"
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见~")
            break

        if not text:
            continue
        if text.lower() in ("exit", "quit", "q"):
            print("再见~")
            break
        if text.startswith("/new"):
            thread_id = f"cli-{int(time.time())}"
            print(f"已开启新会话：{thread_id}")
            continue

        out = app.stream(
            {"messages": [HumanMessage(content=text)]},
            config={"configurable": {"model": settings.DEFAULT_MODEL, "thread_id": thread_id}},
            stream_mode="values"
        )
        #_print_last_ai(out)
        print_stream(out)


