from __future__ import annotations
from typing import Protocol, Dict, Any
import os, datetime as dt

# 可选依赖：按需导入，避免无 MySQL 环境时报错
try:
    import sqlite3
except Exception:
    sqlite3 = None

try:
    import pymysql
except Exception:
    pymysql = None


class LedgerDB(Protocol):
    """数据库仓库统一接口（仅交易表）"""

    def init(self) -> None: ...
    def insert_transaction(self, p: Dict[str, Any]) -> int: ...
    def summarize_transactions(self, plan: Dict[str, Any]) -> Dict[str, Any]: ...


# ---------------------------
# SQLite 实现（仅 transactions）
# ---------------------------
CREATE_SQLITE_TRANSACTIONS = """
CREATE TABLE IF NOT EXISTS transactions(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  occurred_at TEXT NOT NULL,
  item TEXT NOT NULL,
  amount_cents INTEGER NOT NULL,
  currency TEXT NOT NULL,
  type TEXT NOT NULL,
  category TEXT,
  merchant TEXT,
  note TEXT,
  source_message TEXT,
  created_at TEXT NOT NULL
);
"""

INSERT_SQLITE_TRANSACTION = """
INSERT INTO transactions
(occurred_at, item, amount_cents, currency, type, category, merchant, note, source_message, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

class SQLiteLedgerDB:
    def __init__(self, db_path: str = "ledger.db"):
        if sqlite3 is None:
            raise RuntimeError("sqlite3 not available in this environment")
        self.db_path = db_path

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def init(self) -> None:
        with self._conn() as conn:
            conn.execute(CREATE_SQLITE_TRANSACTIONS)

    def insert_transaction(self, p: Dict[str, Any]) -> int:
        created_at = dt.datetime.now().isoformat(timespec="seconds")
        with self._conn() as conn:
            cur = conn.execute(
                INSERT_SQLITE_TRANSACTION,
                (
                    p["occurred_at"],
                    p["item"],
                    int(p["amount_cents"]),
                    p.get("currency", "CNY"),
                    p.get("type", "expense"),
                    p.get("category"),
                    p.get("merchant"),
                    p.get("note"),
                    p.get("source_message"),
                    created_at,
                ),
            )
            return int(cur.lastrowid)

    @staticmethod
    def _start_bound(date_str: str | None) -> str | None:
        if not date_str:
            return None
        if "T" in date_str:
            return date_str
        return f"{date_str}T00:00"

    @staticmethod
    def _end_bound(date_str: str | None) -> str | None:
        if not date_str:
            return None
        if "T" in date_str:
            base = dt.datetime.fromisoformat(date_str.replace("Z", "")) + dt.timedelta(minutes=1)
            return base.isoformat(timespec="minutes")
        end_date = dt.date.fromisoformat(date_str) + dt.timedelta(days=1)
        return f"{end_date.isoformat()}T00:00"

    def summarize_transactions(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        metric = (plan.get("metric") or "sum").lower()
        where: list[str] = []
        params: list[Any] = []

        start_bound = self._start_bound(plan.get("start_iso"))
        if start_bound:
            where.append("occurred_at >= ?")
            params.append(start_bound)

        end_exclusive = plan.get("_end_exclusive")
        end_bound = None
        if end_exclusive:
            end_bound = self._start_bound(end_exclusive)
        else:
            end_bound = self._end_bound(plan.get("end_iso"))
        if end_bound:
            where.append("occurred_at < ?")
            params.append(end_bound)

        keywords = [kw.lower() for kw in plan.get("item_keywords", []) if kw]
        if keywords:
            clause = " OR ".join("LOWER(item) LIKE ?" for _ in keywords)
            where.append(f"({clause})")
            params.extend([f"%{kw}%" for kw in keywords])

        categories = [c for c in plan.get("categories", []) if c]
        if categories:
            placeholders = ",".join("?" for _ in categories)
            where.append(f"category IN ({placeholders})")
            params.extend(categories)

        merchants = [m for m in plan.get("merchants", []) if m]
        if merchants:
            placeholders = ",".join("?" for _ in merchants)
            where.append(f"merchant IN ({placeholders})")
            params.extend(merchants)

        notes = plan.get("notes")
        if notes:
            where.append("LOWER(note) LIKE ?")
            params.append(f"%{str(notes).lower()}%")

        where_sql = " AND ".join(where) if where else "1=1"

        select_cols = ["COUNT(*) AS total_rows"]
        if metric in {"sum", "avg"}:
            select_cols.append("COALESCE(SUM(amount_cents), 0) AS total_cents")
        if metric == "avg":
            select_cols.append("COALESCE(AVG(amount_cents), 0) AS avg_cents")

        agg_sql = f"SELECT {', '.join(select_cols)} FROM transactions WHERE {where_sql}"

        with self._conn() as conn:
            cur = conn.execute(agg_sql, params)
            row = cur.fetchone()
            total_rows = int(row[0]) if row else 0
            total_cents = None
            avg_cents = None
            if metric in {"sum", "avg"} and row and len(row) > 1:
                total_cents = int(row[1])
            if metric == "avg" and row and len(row) > 2:
                avg_cents = float(row[2])

            details: list[Dict[str, Any]] = []
            if metric == "list":
                detail_sql = (
                    f"SELECT occurred_at, item, amount_cents, currency, category, merchant, note "
                    f"FROM transactions WHERE {where_sql} ORDER BY occurred_at ASC"
                )
                dcur = conn.execute(detail_sql, params)
                columns = [desc[0] for desc in dcur.description]
                details = [dict(zip(columns, record)) for record in dcur.fetchall()]

        result: Dict[str, Any] = {
            "status": "ok",
            "metric": metric,
            "total_rows": total_rows,
            "total_cents": total_cents,
            "total_amount_yuan": (total_cents / 100) if total_cents is not None else None,
            "avg_cents": avg_cents,
            "avg_amount_yuan": (avg_cents / 100) if avg_cents is not None else None,
            "details": details,
        }
        if plan.get("time_scope"):
            result["time_scope"] = plan.get("time_scope")
        if start_bound:
            result["start_iso"] = plan.get("start_iso")
        if plan.get("end_iso"):
            result["end_iso"] = plan.get("end_iso")
        return result



# ---------------------------
# MySQL 实现（仅 transactions）
# ---------------------------
CREATE_MYSQL_TRANSACTIONS = """
CREATE TABLE IF NOT EXISTS transactions (
  id INT PRIMARY KEY AUTO_INCREMENT,
  occurred_at DATETIME NOT NULL,
  item VARCHAR(64) NOT NULL,
  amount_cents INT UNSIGNED NOT NULL,
  currency CHAR(3) NOT NULL,
  `type` ENUM('expense','income') NOT NULL,
  category VARCHAR(64) NULL,
  merchant VARCHAR(64) NULL,
  note VARCHAR(255) NULL,
  source_message TEXT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_occurred_at (occurred_at),
  INDEX idx_type_occ (`type`, occurred_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

INSERT_MYSQL_TRANSACTION = """
INSERT INTO transactions
(occurred_at, item, amount_cents, currency, `type`, category, merchant, note, source_message, created_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""



class MySQLLedgerDB:
    def __init__(self):
        if pymysql is None:
            raise RuntimeError("pymysql not available in this environment")
        self.host = os.getenv("MYSQL_HOST", "127.0.0.1")
        self.port = int(os.getenv("MYSQL_PORT", "3306"))
        self.user = os.getenv("MYSQL_USER")
        self.password = os.getenv("MYSQL_PASSWORD")
        self.database = os.getenv("MYSQL_DB")
        self.charset = os.getenv("MYSQL_CHARSET", "utf8mb4")

    def _conn(self):
        return pymysql.connect(
            host=self.host, port=self.port, user=self.user, password=self.password,
            database=self.database, charset=self.charset, cursorclass=pymysql.cursors.DictCursor, autocommit=False
        )

    @staticmethod
    def _to_mysql_dt(iso_str: str) -> str:
        # '2025-08-18T08:00' -> '2025-08-18 08:00:00'
        t = dt.datetime.fromisoformat(iso_str.replace("Z", ""))
        t = t.replace(second=0, microsecond=0)
        return t.strftime("%Y-%m-%d %H:%M:%S")

    def init(self) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_MYSQL_TRANSACTIONS)
            conn.commit()

    def insert_transaction(self, p: Dict[str, Any]) -> int:
        occurred_at_dt = self._to_mysql_dt(p["occurred_at"])
        created_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    INSERT_MYSQL_TRANSACTION,
                    (
                        occurred_at_dt,
                        p["item"],
                        int(p["amount_cents"]),
                        p.get("currency", "CNY"),
                        p.get("type", "expense"),
                        p.get("category"),
                        p.get("merchant"),
                        p.get("note"),
                        p.get("source_message"),
                        created_at,
                    ),
                )
                txn_id = int(cur.lastrowid)
            conn.commit()
        return txn_id

    @staticmethod
    def _start_bound(date_str: str | None) -> str | None:
        if not date_str:
            return None
        if "T" in date_str:
            dt_obj = dt.datetime.fromisoformat(date_str.replace("Z", ""))
        else:
            dt_obj = dt.datetime.fromisoformat(f"{date_str}T00:00")
        return dt_obj.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _end_bound(date_str: str | None) -> str | None:
        if not date_str:
            return None
        if "T" in date_str:
            dt_obj = dt.datetime.fromisoformat(date_str.replace("Z", "")) + dt.timedelta(minutes=1)
        else:
            dt_obj = dt.datetime.fromisoformat(f"{date_str}T00:00") + dt.timedelta(days=1)
        return dt_obj.strftime("%Y-%m-%d %H:%M:%S")

    def summarize_transactions(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        metric = (plan.get("metric") or "sum").lower()
        where: list[str] = []
        params: list[Any] = []

        start_bound = self._start_bound(plan.get("start_iso"))
        if start_bound:
            where.append("occurred_at >= %s")
            params.append(start_bound)

        end_exclusive = plan.get("_end_exclusive")
        end_bound = self._start_bound(end_exclusive) if end_exclusive else self._end_bound(plan.get("end_iso"))
        if end_bound:
            where.append("occurred_at < %s")
            params.append(end_bound)

        keywords = [kw.lower() for kw in plan.get("item_keywords", []) if kw]
        if keywords:
            clause = " OR ".join("LOWER(item) LIKE %s" for _ in keywords)
            where.append(f"({clause})")
            params.extend([f"%{kw}%" for kw in keywords])

        categories = [c for c in plan.get("categories", []) if c]
        if categories:
            placeholders = ",".join(["%s"] * len(categories))
            where.append(f"category IN ({placeholders})")
            params.extend(categories)

        merchants = [m for m in plan.get("merchants", []) if m]
        if merchants:
            placeholders = ",".join(["%s"] * len(merchants))
            where.append(f"merchant IN ({placeholders})")
            params.extend(merchants)

        notes = plan.get("notes")
        if notes:
            where.append("LOWER(note) LIKE %s")
            params.append(f"%{str(notes).lower()}%")

        where_sql = " AND ".join(where) if where else "1=1"

        select_cols = ["COUNT(*) AS total_rows"]
        if metric in {"sum", "avg"}:
            select_cols.append("COALESCE(SUM(amount_cents), 0) AS total_cents")
        if metric == "avg":
            select_cols.append("COALESCE(AVG(amount_cents), 0) AS avg_cents")

        agg_sql = f"SELECT {', '.join(select_cols)} FROM transactions WHERE {where_sql}"

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(agg_sql, params)
                row = cur.fetchone()
                total_rows = int(row["total_rows"]) if row else 0
                total_cents = None
                avg_cents = None
                if metric in {"sum", "avg"} and row and "total_cents" in row:
                    total_cents = int(row["total_cents"])
                if metric == "avg" and row and "avg_cents" in row:
                    avg_cents = float(row["avg_cents"])

                details: list[Dict[str, Any]] = []
                if metric == "list":
                    detail_sql = (
                        f"SELECT occurred_at, item, amount_cents, currency, category, merchant, note "
                        f"FROM transactions WHERE {where_sql} ORDER BY occurred_at ASC"
                    )
                    cur.execute(detail_sql, params)
                    details = cur.fetchall()

        result: Dict[str, Any] = {
            "status": "ok",
            "metric": metric,
            "total_rows": total_rows,
            "total_cents": total_cents,
            "total_amount_yuan": (total_cents / 100) if total_cents is not None else None,
            "avg_cents": avg_cents,
            "avg_amount_yuan": (avg_cents / 100) if avg_cents is not None else None,
            "details": details,
        }
        if plan.get("time_scope"):
            result["time_scope"] = plan.get("time_scope")
        if plan.get("start_iso"):
            result["start_iso"] = plan.get("start_iso")
        if plan.get("end_iso"):
            result["end_iso"] = plan.get("end_iso")
        return result



# ---------------------------
# 工厂：按环境选择方言
# ---------------------------
def get_db() -> LedgerDB:
    dialect = os.getenv("DB_DIALECT", "sqlite").lower()
    if dialect == "mysql":
        return MySQLLedgerDB()
    # sqlite 默认路径可用 SQLITE_PATH 环境变量覆盖
    db_path = os.getenv("SQLITE_PATH", "ledger.db")
    return SQLiteLedgerDB(db_path=db_path)
