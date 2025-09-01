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
