# Custom messenge types for the application
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import streamlit as st


class HumanMessage:
    """
    This class is used to represent a message from a human.
    When user want to find a vocab.
    """

    def __init__(self, content: str):
        self.content = content
    
    def render(self):
        with st.chat_message("Question"):
            st.markdown(self.content)


# class MaziiApiMessage:
#     """
#     This class is used to represent a message from Mazii API with markdown style.
#     It will be ignored when sending to llm.
#     """

#     def __init__(self, api_response: dict, vocab: str, parent=None):
#         self.parent = parent
#         self.vocab = vocab
#         self.phonetic = api_response["phonetic"]
#         self.means = api_response["means"]
        
#         self.phonetic_msg = f"*Phoentic*: *{self.phonetic}*\n\n"
#         self.example_msgs = []
#         for item in self.means:
#             mean = item["mean"]
#             examples = item["examples"]
#             content = ""
#             content += "---\n"
#             content += f"*Mean*: {mean}\n\n"
#             if examples is not None:
#                 for example in examples:
#                     content += f"{example['content']}\n\n"
#                     # content += f"{example['transcription']}\n\n"
#                     content += f"{example['mean']}\n\n"
#             self.example_msgs.append(content)
        
#         self.content = self.phonetic_msg + "\n".join(self.example_msgs)
    
#     def render(self, get_per_meaning_component=None, get_final_component=None, idx=None, conversation = None):
#         with st.chat_message("mazii"):
#             st.markdown(self.phonetic_msg)
            
#             for mean_idx, (content, example_dict) in enumerate(zip(self.example_msgs, self.means)):
#                 st.markdown(content)
#                 if get_per_meaning_component is not None:
#                     comp_idx = f"{idx}-{mean_idx}"
#                     _ = get_per_meaning_component(comp_idx, self.vocab, example_dict)
            
#             if get_final_component is not None:
#                 _ = get_final_component(self, idx, self.vocab, self.phonetic, self.means, conversation, self.content)


class ChatExampleMessage(AIMessage):
    """
    This class is used to represent a message from the chatbot to show an example.
    """

    def __init__(self, content: str, parent=None):
        super().__init__(content=content)
        self.parent = parent

    def render(self):
        with st.chat_message("assistant"):
            st.markdown(self.content)


# class ChatExplainMessage(AIMessage):
#     """
#     This class is used to represent a message from the chatbot to explain the answer.
#     """

#     def __init__(self, content: str, parent=None):
#         super().__init__(content=content)
#         self.parent = parent
    
#     def render(self):
#         with st.chat_message("assistant"):
#             st.markdown(self.content)


if __name__ == "__main__":
    example_message = ChatExampleMessage(content="This is the example message")
    example_message.render()
