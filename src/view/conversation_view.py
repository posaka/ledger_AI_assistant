import streamlit as st
from agents.expense_tracker_agent import app, _print_last_ai
from agents.utils.user_profile import memobase_client
from langchain_core.messages import HumanMessage
st.title("智能记账助手")

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