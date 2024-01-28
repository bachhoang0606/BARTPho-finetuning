import os

import streamlit as st
from langchain.schema import AIMessage

from llm_sender import ChatMessageSender
from messages import ChatExampleMessage, HumanMessage
from api import __model_list
from env import load_dotenv_if_exists


_init_messages = []


def init_page():
    st.set_page_config(page_title="Japanese dictionary", page_icon="📚")
    st.header("Enter word in Japanese")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = _init_messages


def select_model():
    model_name = st.sidebar.radio(
        "Choose LLM: (require refresh after change)", __model_list
    )
    return model_name

def main():
    load_dotenv_if_exists()
    _user_id = os.getenv("EXAMPLE_USER_ID")
    assert _user_id, "Please set EXAMPLE_USER_ID in .env file"

    init_page()
    llm = select_model()
    sender = ChatMessageSender(llm, mongo_user_id=_user_id)
    init_messages()

    # # Supervise user input
    if user_input := st.chat_input("Input your question!"):
        human_msg = HumanMessage(content=user_input)
        st.session_state.messages.append(human_msg)
        message = sender.send_get_legal_message(question=human_msg.content)

    # Display chat history
    messages = st.session_state.get("messages", [])
    for idx, message in enumerate(messages):
        if isinstance(message, ChatExampleMessage):
            message.render()
        elif isinstance(message, HumanMessage):
            message.render()
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        else: 
            st.warning(f"Unhandled message type: {type(message)}")


if __name__ == "__main__":
    main()
