from pprint import pprint
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from env import load_dotenv_if_exists
import os
from peft import PeftModel

load_dotenv_if_exists()
HF_TOKEN = os.getenv("ACCESS_HUGGINGFACE_TOKEN")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "bkai-foundation-models/vietnamese-llama2-7b-120GB"
PEFT_MODEL = "Bachhoang/vietnamese-llama2-7b-120GB-legal"

"""
##### Câu hỏi: Nội dung của Điều 14 3380/QĐ-UBND là gì? ### Trả lời: 
Điều 14 3380/QĐ-UBND về việc phê duyệt điều lệ (sửa đổi, bổ sung) hội 
khoa học kỹ thuật xây dựng khánh hòa Ban Chấp hành có thể biểu quyết 
bằng hình thức giơ tay hoặc bỏ phiếu kín. Việc quy định hình thức biểu 
quyết do Ban Chấp hành quyết định; d) Các nghị quyết, quyết định của Ban 
Chấp hành được thông qua khi có trên 2/3 (hai phần ba) tổng số ủy viên Ban Chấp 
hành dự họp biểu quyết tán thành. Trong trường hợp số ý kiến tán thành và không 
tán thành ngang nhau thì quyết định thuộc về bên có ý kiến của Chủ tịch Hội. #####
"""

def create_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
        token=HF_TOKEN,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN,)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

def generate_prompt(question: str) -> str:
    return f"""<s>[INST] Câu hỏi: {question} [/INST] """.strip()


def inference(model, text: str):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    inputs_length = len(inputs["input_ids"][0])
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.0001)
    return tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)

question = ""

# load model
model, tokenizer = create_model_and_tokenizer()

# infrence fine tune model MODEL_NAME
model = PeftModel.from_pretrained(model, PEFT_MODEL)

def inference_llama2(text : str):
    return inference(model, generate_prompt(text))

if __name__ == "__main__":

    summary = inference(model, generate_prompt("Nội dung của Điều 14 3380/QĐ-UBND là gì?"))
    pprint(summary)
