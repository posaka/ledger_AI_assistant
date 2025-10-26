import time

import streamlit as st
from agents.expense_tracker_agent import app, settings, _print_last_ai, DB
from agents.utils.user_profile import memobase_client

def _ensure_message_buffer() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _reset_auth_state() -> None:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.user_id = None
    st.session_state.config = None
    st.session_state.thread_id = None
    st.session_state.messages = []


def _logout() -> None:
    _reset_auth_state()
    st.rerun()


def _render_auth_panel() -> None:
    st.subheader("登录或注册账号")
    login_tab, register_tab = st.tabs(["登录", "注册"])

    with login_tab:
        with st.form("login_form"):
            login_username = st.text_input("账号", key="login_username")
            login_password = st.text_input("密码", type="password", key="login_password")
            login_submit = st.form_submit_button("登录")

        if login_submit:
            username = login_username.strip()
            password = login_password
            if not username or not password:
                st.error("请输入账号和密码。")
            else:
                try:
                    user_id = DB.authenticate_user(username, password)
                except Exception as exc:
                    st.error(f"登录失败：{exc}")
                else:
                    if user_id:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.user_id = user_id
                        st.session_state.thread_id = f"{username}-{int(time.time())}"
                        st.session_state.config = {
                            "configurable": {
                                "model": settings.DEFAULT_MODEL,
                                "thread_id": st.session_state.thread_id,
                                "user_id": user_id,
                            }
                        }
                        st.session_state.messages = []
                        st.rerun()
                    else:
                        st.error("账号或密码不正确。")

    with register_tab:
        with st.form("register_form"):
            register_username = st.text_input("账号", key="register_username")
            register_password = st.text_input("密码", type="password", key="register_password")
            register_confirm = st.text_input("确认密码", type="password", key="register_password_confirm")
            register_submit = st.form_submit_button("注册")

        if register_submit:
            username = register_username.strip()
            password = register_password
            confirm = register_confirm

            if not username:
                st.error("账号不能为空。")
            elif len(username) < 3:
                st.error("账号长度至少 3 位。")
            elif not password:
                st.error("密码不能为空。")
            elif len(password) < 6:
                st.error("密码长度至少 6 位。")
            elif password != confirm:
                st.error("两次输入的密码不一致。")
            else:
                try:
                    if DB.username_exists(username):
                        st.error("账号已存在，请直接登录。")
                        return
                    memo_user_id = memobase_client.add_user()
                    DB.register_user(username, password, memo_user_id)
                    st.success("注册成功，请使用账号登录。")
                except Exception as exc:
                    st.error(f"注册失败：{exc}")


_ensure_message_buffer()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    _render_auth_panel()
    st.stop()

with st.sidebar:
    st.markdown(f"**当前账号：** {st.session_state.username}")
    if st.button("退出登录", key="logout_button"):
        _logout()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

pages = [
    st.Page(".\conversation_view.py", title="智能助手"),
    st.Page(".\ledger_view.py", title="账本"),
]
pg = st.navigation(pages)
pg.run()

