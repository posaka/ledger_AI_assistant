import json
import datetime as dt
from typing import Literal

LOG_PATH = "chat_history.jsonl"  # 记录会存到当前目录这个文件

import datetime as dt

def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_msg(role: Literal["user", "assistant"], text: str):
    record = {
        "role": role,
        "text": text,
        "timestamp": _utc()
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    