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

def format_messages(messages):
    formatted_messages = []
    for message in messages:
        if isinstance(message, HumanMessage):
            formatted_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, SystemMessage):
            formatted_messages.append({"role": "system", "content": message.content})
        elif isinstance(message, AIMessage):
            formatted_messages.append({"role": "assistant", "content": message.content})
    return formatted_messages

# return the first user's id in the userlist; if there are no users, create one.
def init_users():
    users = memobase_client.get_all_users()
    if not users:
        uid = memobase_client.add_user()
        return uid
    return users[0]["id"]