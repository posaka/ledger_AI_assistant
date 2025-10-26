from memobase import MemoBaseClient
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage, ToolMessage
from dotenv import load_dotenv
import os

load_dotenv()

MEMOBASE_URL = os.environ.get("MEMOBASE_URL")
MEMOBASE_SECRET = os.environ.get("MEMOBASE_SECRET")

memobase_client = MemoBaseClient(
    project_url=MEMOBASE_URL,
    api_key=MEMOBASE_SECRET,
)

user = memobase_client.get_user(memobase_client.get_all_users()[0]["id"])

print(user.context())