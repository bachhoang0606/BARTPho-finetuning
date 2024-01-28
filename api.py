import os

import requests
from env import load_dotenv_if_exists

load_dotenv_if_exists()
_hf_token = os.getenv("ACCESS_HUGGINGFACE_TOKEN")
assert _hf_token, "Please set ACCESS_HUGGINGFACE_TOKEN in .env file"

GPT2_API_URL = "https://api-inference.huggingface.co/models/Bachhoang/gpt2-vietnamese-legal"
BART_API_URL = "https://api-inference.huggingface.co/models/Bachhoang/BARTpho-vietnamese-legal"

headers = {"Authorization": f"Bearer {_hf_token}"}

__model_list = (
    "BARTpho-vietnamese-legal",
    "gpt2-vietnamese-legal",
)

__model_url = {
    "BARTpho-vietnamese-legal": BART_API_URL,
    "gpt2-vietnamese-legal": GPT2_API_URL,
}

def get_answer(model_name, payload):
    assert model_name in __model_list, f"Model name must be one of {__model_list}"

    response = requests.post(__model_url[model_name], headers=headers, json=payload)
    data = response.json()
    if "error" not in data:
        data = data[0]["generated_text"]
    else:
        data = data["error"]
        
    return data, payload

if __name__ == "__main__":
    from env import load_dotenv_if_exists
    from prompt import legal_prompt, create_conversation


    load_dotenv_if_exists()

    print(__model_url)

    # prompts message
    message = "Nội dung của Điều 23 01/2013/TT-UBDT là gì?"
    legal_message = create_conversation({"question":message}, legal_prompt)

    output, payload = get_answer(model_name="gpt2-vietnamese-legal", payload=legal_message)
    # print answer
    print(output)
