from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.store.postgres import PostgresStore
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from psycopg import Connection
import os

index = {
    "dims": 1536,
    "embed": HuggingFaceEmbeddings(),
}



class PgStoreHandle:
    def __init__(self, conn_str, index=None):
        self._conn = Connection.connect(conn_str, autocommit=True)
        self.store = PostgresStore(self._conn, index=index)
        self.store.setup()

    def close(self):
        self._conn.close()

def get_store() -> BaseStore:
    dialect = os.getenv("STORE_DIALECT", "memory").lower()
    if dialect == "postgres":
        conn_str = os.getenv("PG_CONN_STR")
        if not conn_str:
            raise ValueError("PG_CONN_STR environment variable is required for Postgres store")
        print("Using Postgres store")
        return PgStoreHandle(conn_str).store
    else:
        print("Using In-Memory store")
        return InMemoryStore(index=index)