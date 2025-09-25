import io
import requests
import matplotlib.pyplot as plt
from PIL import Image
from langgraph.graph.state import CompiledStateGraph  # 仅为类型提示，可不写

def draw_graph(app: CompiledStateGraph,
                         fmt: str = "png",
                         base_url: str = "https://kroki.io",
                         timeout: int = 30,
                         save_path: str | None = None):
    """
    先拿 Mermaid 源，再用 Kroki 渲染并直接显示。
    fmt: 'png' | 'svg' | 'pdf'（此函数内直接展示仅支持 png）
    """
    assert fmt in {"png", "svg", "pdf"}
    mmd = app.get_graph().draw_mermaid()

    # Kroki: POST /mermaid/{format}
    url = f"{base_url}/mermaid/{fmt}"
    resp = requests.post(url, data=mmd.encode("utf-8"),
                         headers={"Content-Type": "text/plain"},
                         timeout=timeout)
    resp.raise_for_status()
    data = resp.content

    # 可选落盘
    if save_path:
        with open(save_path, "wb") as f:
            f.write(data)

    # 直接打开（仅 png 能被 PIL 直接识别 & 用 matplotlib 显示）
    if fmt == "png":
        img = Image.open(io.BytesIO(data))
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    else:
        print(f"{fmt} 已生成（{len(data)} bytes）——该格式不在此函数内直接展示。")

