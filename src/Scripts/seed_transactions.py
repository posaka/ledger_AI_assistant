# 使用命令
# python src/scripts/seed_transactions.py --username demo --password Demo123! --count 300

from __future__ import annotations

import argparse
import random
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from agents.utils.db_repo import get_db
except ImportError as exc:  # pragma: no cover - safety guard for script execution
    raise SystemExit(f"无法导入数据库仓库模块：{exc}") from exc

try:
    from agents.utils.user_profile import memobase_client
except Exception:  # pragma: no cover - memobase 组件可选
    memobase_client = None

NOW = datetime.now()
ONE_YEAR_AGO = NOW - timedelta(days=365)

EXPENSE_DATA: Dict[str, Dict[str, List[str]]] = {
    "餐饮": {
        "items": ["早餐", "午餐", "外卖", "咖啡", "奶茶", "宵夜", "甜品", "水果"],
        "merchants": ["美团外卖", "星巴克", "瑞幸咖啡", "连咖啡", "喜茶", "海底捞", "奈雪"],
    },
    "交通": {
        "items": ["地铁", "打车", "高铁", "滴滴", "共享单车", "加油"],
        "merchants": ["滴滴出行", "哈啰单车", "中国石化"],
    },
    "购物": {
        "items": ["日用品", "衣服", "图书", "数码配件", "护肤品", "家居用品"],
        "merchants": ["淘宝", "京东", "拼多多", "当当", "小米之家"],
    },
    "娱乐": {
        "items": ["电影票", "游戏点卡", "会员订阅", "KTV", "剧本杀"],
        "merchants": ["猫眼电影", "腾讯视频", "爱奇艺", "Spotify", "网易游戏"],
    },
    "健康": {
        "items": ["健身房", "瑜伽课", "按摩", "药品", "体检"],
        "merchants": ["Keep", "超级猩猩", "华润药店"],
    },
}

INCOME_DATA: Dict[str, Dict[str, List[str]]] = {
    "工资": {
        "items": ["月薪", "季度奖金", "年终奖"],
        "merchants": ["公司打款"],
    },
    "理财": {
        "items": ["基金收益", "股票分红", "利息收入"],
        "merchants": ["招商银行", "蚂蚁财富", "华泰证券"],
    },
    "其他": {
        "items": ["红包", "转账", "兼职收入", "二手交易"],
        "merchants": ["微信", "支付宝", "闲鱼"],
    },
}

EXPENSE_NOTES = [
    "和朋友聚餐",
    "下班太晚点了份外卖",
    "买了点小零食犒劳自己",
    "通勤路上喝杯咖啡醒醒脑",
    "换季了添置衣服",
    "周末补了节电影",
    "健身后蛋白补给",
    "",
    "",
]

INCOME_NOTES = [
    "本月工资到账",
    "基金小赚一笔",
    "收到亲戚红包",
    "做了个小兼职",
    "",
]


def _random_datetime() -> datetime:
    delta_seconds = random.randint(0, int((NOW - ONE_YEAR_AGO).total_seconds()))
    occurred = ONE_YEAR_AGO + timedelta(seconds=delta_seconds)
    return occurred.replace(second=0, microsecond=0)


def _random_amount(txn_type: str) -> float:
    if txn_type == "income":
        return round(random.uniform(200, 8000), 2)
    return round(random.uniform(5, 800), 2)


def _build_payload(user_id: str, txn_type: str) -> Dict[str, Any]:
    data_map = INCOME_DATA if txn_type == "income" else EXPENSE_DATA
    category = random.choice(list(data_map.keys()))
    item = random.choice(data_map[category]["items"])
    merchants = data_map[category].get("merchants") or []
    merchant = random.choice(merchants) if merchants and random.random() < 0.8 else None
    amount = _random_amount(txn_type)
    occurred_at = _random_datetime()

    note_pool = INCOME_NOTES if txn_type == "income" else EXPENSE_NOTES
    note = random.choice(note_pool)

    payload: Dict[str, Any] = {
        "user_id": user_id,
        "type": txn_type,
        "item": item,
        "category": category,
        "merchant": merchant,
        "amount_cents": int(round(amount * 100)),
        "currency": "CNY",
        "occurred_at": occurred_at.isoformat(timespec="minutes"),
        "note": note or None,
        "source_message": f"{occured_desc(occurred_at)} {item} 花了 {amount:.2f} 元",
    }
    return payload


def occured_desc(occurred_at: datetime) -> str:
    return occurred_at.strftime("%Y年%m月%d日 %H:%M")


def ensure_user(db, username: str, password: str, requested_user_id: str | None) -> Tuple[str, bool]:
    """返回 user_id，第二个返回值表示是否新建。"""
    if db.username_exists(username):
        user_id = db.authenticate_user(username, password)
        if not user_id:
            raise SystemExit("账号已存在但密码不正确，请确认后再试。")
        return user_id, False

    user_id = requested_user_id or create_memobase_user()
    try:
        db.register_user(username, password, user_id)
    except Exception as exc:
        raise SystemExit(f"注册账号失败：{exc}") from exc
    return user_id, True


def create_memobase_user() -> str:
    if memobase_client is None:
        return str(uuid.uuid4())
    try:
        return memobase_client.add_user()
    except Exception:
        return str(uuid.uuid4())


def seed_transactions(
    username: str,
    password: str,
    count: int,
    seed: int | None,
    user_id: str | None,
) -> None:
    if seed is not None:
        random.seed(seed)

    db = get_db()
    db.init()

    user_id_value, created = ensure_user(db, username, password, user_id)

    types = ["expense", "expense", "expense", "income"]
    inserted = 0
    for _ in range(count):
        txn_type = random.choice(types)
        payload = _build_payload(user_id_value, txn_type)
        payload_copy = dict(payload)
        payload_copy.pop("user_id", None)
        db.insert_transaction(user_id_value, payload_copy)
        inserted += 1

    print(f"✅ 完成：账号 {username}（user_id={user_id_value}）新增 {inserted} 条交易记录。")
    if created:
        print("ℹ️ 该账号为脚本自动创建。如需在前端登录，请使用脚本填写的账号密码。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="向 ledger.db 写入随机测试数据（涵盖近一年）。",
    )
    parser.add_argument("--username", required=True, help="用于写入数据的账号用户名。若不存在将自动创建。")
    parser.add_argument("--password", required=True, help="账号密码。若新建账号将使用该密码。")
    parser.add_argument("--count", type=int, default=200, help="要插入的交易数量，默认 200。")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，便于复现。")
    parser.add_argument("--user-id", dest="user_id", default=None, help="绑定的 MemoBase user_id，不传则自动创建/生成。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_transactions(
        username=args.username.strip(),
        password=args.password,
        count=args.count,
        seed=args.seed,
        user_id=args.user_id,
    )


if __name__ == "__main__":
    main()
