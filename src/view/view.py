import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agents.expense_tracker_agent import app, settings, _print_last_ai
from agents.utils.user_profile import init_users, memobase_client

st.title("智能账本")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    import time
    st.session_state.thread_id = f"demo-{int(time.time())}"

if "config" not in st.session_state:
    uid = init_users()
    st.session_state.config = {
        "configurable": {
            "model": settings.DEFAULT_MODEL,
            "thread_id": st.session_state.thread_id,
            "user_id": uid
        }
    }

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_input := st.chat_input("请输入："):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("账账发力中..."):
            response = app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=st.session_state.config
            )
        st.write(response["messages"][-1].content)
        st.session_state.messages.append({"role": "assistant", "content": response["messages"][-1].content})
        _print_last_ai(response)
        print(memobase_client.get_all_users())