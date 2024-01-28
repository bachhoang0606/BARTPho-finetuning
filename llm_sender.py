"""
LLM Sender and related components.
"""

import streamlit as st
from api import get_answer
from messages import ChatExampleMessage, HumanMessage
from langchain.schema import AIMessage
from prompt import common_prompt, legal_prompt, create_conversation
from vocabulary_controller import add_vocabulary

class ChatMessageSender:
    """
    Send message to llm and add the answer to the chat.
    """

    def __init__(self, llm, mongo_user_id=None):
        self.llm = llm
        self.user_id = mongo_user_id

    def send_get_legal_message(self, question: str):
        with st.spinner(f"{self.llm} is typing ..."):
            generated_text, question = get_answer(
                self.llm,
                create_conversation(
                    {
                        "question": question,
                    },
                    legal_prompt,
                ),
            )
        st.session_state.messages.append(ChatExampleMessage(content=generated_text))

    def send_get_common_message(self, question: str):
        with st.spinner(f"{self.llm} is typing ..."):
            generated_text, question = get_answer(
                self.llm,
                create_conversation(
                    {
                        "question": question,
                    },
                    common_prompt,
                ),
            )
        st.session_state.messages.append(ChatExampleMessage(content=generated_text))

    # def get_save_btn(self, owner_message: HumanMessage, idx: str, vocab: str, phonetic: str, means: dict, conversation: list=None, content=None):
    #     """
    #     Callback for explain and save buttons
    #     """
    #     # saved = st.session_state.get(f"save-btn-{idx}-saved", False)
    #     # if not saved:
    #     if st.button("Save", key=f"save-btn-{idx}"):
    #         st.session_state[f"save-btn-{idx}-saved"] = True
    #         # print(f"Saved: vocab: {vocab}, phonetic: {phonetic}, means: {means}")
    #         other_messages = []
    #         if conversation is not None:
    #             # find index of owner_message
    #             owner_idx = conversation.index(owner_message)
    #             # end index is the next message of instance of HumanMessage
    #             end_idx = len(conversation) - 1
    #             for _idx in range(owner_idx, len(conversation)):
    #                 message = conversation[_idx]
    #                 if isinstance(message, HumanMessage):
    #                     end_idx = _idx - 1
    #                     break
    #             other_messages = conversation[owner_idx + 1 : end_idx + 1]
    #             other_messages = [message.content for message in other_messages]
    #             print(other_messages)
    #         add_vocabulary(self.user_id, {
    #             "vocabulary": vocab,
    #             # "phonetic": phonetic,
    #             "example": content,
    #             "conversation": other_messages,
    #         })
    #         print(f"Saved: vocab: {vocab}, ...")
    #     # else:
    #     #     st.button("Saved", key=f"save-btn-{idx}", disabled=True)


if __name__ == "__main__":
    from env import load_dotenv_if_exists
    from Translate import init_messages

    load_dotenv_if_exists()
    # get llm
    llm = "BARTpho-vietnamese-legal"
    init_messages()

    sender = ChatMessageSender(llm)

    question = "Nội dung của Điều 23 01/2013/TT-UBDT là gì?"
    message = HumanMessage(content=question)

    _init_messages = [
        message,
    ]
    if not st.session_state.messages:
        st.session_state.messages = _init_messages

    for idx, message in enumerate(st.session_state.messages):
        if isinstance(message, ChatExampleMessage):
            message.render()
        elif isinstance(message, HumanMessage):
            message.render()
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
