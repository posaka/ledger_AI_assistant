import io
import matplotlib.pyplot as plt
from PIL import Image
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph.state import CompiledStateGraph


def draw_graph(agent: CompiledStateGraph):
    # 渲染成 PNG（优先本地，失败则用 Mermaid 在线）
    try:
        png_bytes = agent.get_graph().draw_png()
    except:
        png_bytes = agent.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API
        )

    # 显示弹窗
    img = Image.open(io.BytesIO(png_bytes))
    plt.imshow(img)
    plt.axis("off")
    plt.show()