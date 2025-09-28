from src.agents.utils.store import get_store
import os
from dotenv import load_dotenv
load_dotenv()

def print_postgres():
    store = get_store()
    print(store.search(("memories",)))

if __name__ == "__main__":
    print_postgres()