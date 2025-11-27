import os
from dotenv import load_dotenv
import requests
from typing import List, Dict, Optional
from core.llm import ModelT
from langchain_core.messages import BaseMessage, SystemMessage

load_dotenv()

BAIDU_API_KEY = os.getenv("BAIDU_API_KEY")
DEFAULT_BAIDU_URL = os.getenv(
    "BAIDU_API_URL", "https://qianfan.baidubce.com/v2/ai_search/web_search"
)

def search_baidu(
    query: str,
    top_k: int = 10,
    recency: Optional[str] = None,
    sites: Optional[List[str]] = None,
    api_key: Optional[str] = BAIDU_API_KEY,
    api_url: Optional[str] = DEFAULT_BAIDU_URL,
    timeout: int = 10,
) -> Dict:
    """
    调用百度千帆 web_search 接口并返回统一结果结构。

    返回结构示例:
    {
        "request_id": "...",
        "results": [
            {"title": "...", "url": "...", "snippet": "...", "date": "...", "source": "..."},
            ...
        ],
        "raw": <原始响应 JSON>
    }

    参数:
    - query: 查询文本（必填）
    - top_k: 最多返回条数 (网页 top_k 最大 50)
    - recency: 可选时效过滤, 如 "week","month","year" 或符合文档的其它值
    - sites: 可选站点白名单列表
    - api_key: 可选覆盖环境变量的 AppBuilder API Key
    - api_url: 可选覆盖默认 endpoint
    """
    if not query or not query.strip():
        raise ValueError("query 不能为空")

    api_key = api_key or os.getenv("BAIDU_API_KEY")
    if not api_key:
        raise ValueError("缺少 BAIDU_API_KEY（AppBuilder API Key），请在环境变量中设置或通过参数传入）")

    url = api_url
    payload = {
        "messages": [{"role": "user", "content": query}],
        "search_source": "baidu_search_v2",
        "resource_type_filter": [{"type": "web", "top_k": min(max(1, top_k), 50)}],
    }
    if recency:
        payload["search_recency_filter"] = recency
    if sites:
        payload["search_filter"] = {"match": {"site": sites}}

    headers = {
        "Content-Type": "application/json",
        "X-Appbuilder-Authorization": f"Bearer {api_key}",
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    refs = data.get("references") or data.get("references", []) or []
    results = []
    for r in refs[:top_k]:
        results.append({
            "title": r.get("title"),
            "url": r.get("url") or r.get("link"),
            "snippet": r.get("content") or r.get("web_anchor") or "",
            "date": r.get("date"),
            "source": r.get("website") or r.get("web_anchor")
        })

    return {
        "request_id": data.get("request_id") or data.get("requestId"),
        "results": results,
        "raw": data,
    }

def generate_recommendation(messages: list, llm: ModelT) -> str:
    """
    基于提供的信息，搜索网页生成推荐内容
    """
    topic_prompt = "请根据以下对话内容，提取其中多次出现，且在本条消息的上一条消息中提到过的主题，生成一个用于搜索的主题短语。生成内容不能为空"
    topic = llm.invoke(messages + [SystemMessage(content=topic_prompt)])
    search_results = search_baidu(topic.content, top_k=5)
    recommendation_prompt = f"""
    请根据搜索主题：{topic.content}
    以及搜索结果：{search_results['results']}
    生成一个简短的推荐内容。推荐内容应严格按照搜索结果相关条目填入信息，显示3~5条相关内容。
    具体格式如下：
    <一句简短的开头语>，
    1. <网页标题（截取至20字内，超出用省略号代替）>：<网页链接>
    <网页snippet>
    2. <网页标题（截取至20字内，超出用省略号代替）>：<网页链接>
    <网页snippet>
    ...
    
    例：对于主题<美食推荐>，生成以下推荐内容：
    我发现你最近对美食很感兴趣，以下是一些美食的推荐哦：
    
    《12 月潮汕家庭游攻略 ...》：https://www.163.com/dy/article/KF9N7BOA0556CVM1.html
    这里既有南澳岛碧海蓝天的治愈风光...更有牛肉火锅、砂锅粥、腐乳饼等让人舌尖生津的地道美食。
    
    《中国十大名菜》：https://baike.baidu.com/item/中国十大名菜/9504492
    北京烤鸭源于明代御膳,以果木炭烤制为特色;东坡肉相传由北宋苏东坡改良烹饪技法,以酒焖炖酥烂不碎...
    
    《武汉八大名吃》：https://baike.baidu.com/item/武汉八大名吃/2670979
    蔡林记热干面武汉热干面与山西的刀削面、两广的伊府面、四川的担担面、北方的炸酱面并称为中国五大名面...
    """
    recommendation = llm.invoke(messages + [SystemMessage(content=recommendation_prompt)])
    return recommendation.content