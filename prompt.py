from langchain.prompts.prompt import PromptTemplate

common_prompt = PromptTemplate(
    input_variables=["question"],
    template="{question}",
)

legal_prompt = PromptTemplate(
    input_variables=["question"],
    template="##### Câu hỏi: {question} ### Trả lời:",
    # template="<startofstring> Câu hỏi: {question} ### Trả lời:",
)

def create_conversation(question, prompt_template):
    prompt = prompt_template.format(**question)
    return prompt

if __name__ == "__main__":
   
    legal_prompt = legal_prompt.format(**{"question": "Nội dung của Điều 23 01/2013/TT-UBDT là gì?"})
    common_prompt = common_prompt.format(**{"question": "Nội dung của Điều 23 01/2013/TT-UBDT là gì?"})
    print(legal_prompt)
    print(common_prompt)
