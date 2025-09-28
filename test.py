import os
import sys
import psycopg

# 从环境变量取，没设就用你给的默认值
conn_str = os.getenv(
    "PG_CONN_STR",
    "postgresql://postgres:pass@localhost:15432/postgres",
)

try:
    # autocommit=True 方便只跑测试查询
    with psycopg.connect(conn_str, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            one = cur.fetchone()[0]
            # 取服务器版本
            cur.execute("SHOW server_version;")
            version = cur.fetchone()[0]
            print(f"OK: SELECT 1 -> {one}, server_version={version}")
    sys.exit(0)
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
    sys.exit(1)
