import streamlit as st
from langgraph_backend import (
    chatbot,
    get_all_chats,
    save_chat_title,
)
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import time

#  Utility Functions 

def generate_thread_id():
    return str(uuid.uuid4())

def generate_chat_title(text: str, max_len: int = 40) -> str:
    text = text.strip().replace("\n", " ")
    return text[:max_len] + ("..." if len(text) > max_len else "")

def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])

#  Session Setup 

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = None

if "pending_new_chat" not in st.session_state:
    st.session_state["pending_new_chat"] = False

#  Sidebar UI 

st.sidebar.title("LangGraph Chatbot")

# New chat (lazy creation)
if st.sidebar.button("➕ New Chat"):
    st.session_state["pending_new_chat"] = True
    st.session_state["message_history"] = []

st.sidebar.header("My Conversations")

for thread_id, title in get_all_chats():
    if st.sidebar.button(title, key=f"thread_{thread_id}"):
        st.session_state["thread_id"] = thread_id
        st.session_state["pending_new_chat"] = False

        messages = load_conversation(thread_id)
        history = []

        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            history.append({"role": role, "content": msg.content})

        st.session_state["message_history"] = history

#  Main UI 

# Render conversation history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type here...")

#  Streaming Logic 

if user_input:
    # Create thread ONLY when first message is sent
    if st.session_state["pending_new_chat"] or st.session_state["thread_id"] is None:
        st.session_state["thread_id"] = generate_thread_id()
        st.session_state["pending_new_chat"] = False

    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    # Save & render user message
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    #  Assistant Streaming 

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        buffer = ""
        cursor_visible = True

        with st.status("Thinking…", state="running") as status:
            status.write("Understanding your question…")
            time.sleep(0.3)

            status.write("Generating response…")
            time.sleep(0.3)

            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if not isinstance(message_chunk, AIMessage):
                    continue

                if not message_chunk.content:
                    continue

                buffer += message_chunk.content

                if buffer.endswith(("\n", " ", ".", "!", "?", "`")):
                    full_response += buffer
                    buffer = ""

                    cursor = "▍" if cursor_visible else ""
                    placeholder.markdown(full_response + cursor)
                    cursor_visible = not cursor_visible

                    time.sleep(0.015)

            if buffer:
                full_response += buffer
                placeholder.markdown(full_response)

            status.update(label="Response ready", state="complete")

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": full_response}
    )

    #  Persist title to SQLite (safe, once per thread)
    save_chat_title(
        st.session_state["thread_id"],
        generate_chat_title(full_response),
    )
