import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  
# python3 call_model/PhoGPT_api.py

acess_token = "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"

model_path = 'vinai/PhoGPT-7B5-Instruct'  
  
config = AutoConfig.from_pretrained(model_path, token=acess_token)  
# config.init_device = "cuda"
# config.attn_config['attn_impl'] = 'triton' # Enable if "triton" installed!
  
model = AutoModelForCausalLM.from_pretrained(  
    model_path, config=config, torch_dtype=torch.bfloat16, token=acess_token  
)
# If your GPU does not support bfloat16:
# model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, trust_remote_code=True)
model.eval()  
  
tokenizer = AutoTokenizer.from_pretrained(model_path, token=acess_token)  
  
PROMPT_TEMPLATE = "### Câu hỏi:\n{instruction}\n\n### Trả lời:"  

# Some instruction examples
# instruction = "Viết bài văn nghị luận xã hội về {topic}"
# instruction = "Viết bản mô tả công việc cho vị trí {job_title}"
# instruction = "Sửa lỗi chính tả:\n{sentence_or_paragraph}"
# instruction = "Dựa vào văn bản sau đây:\n{text}\nHãy trả lời câu hỏi: {question}"
# instruction = "Tóm tắt văn bản:\n{text}"


instruction = "Viết bài văn nghị luận xã hội về an toàn giao thông"
# instruction = "Sửa lỗi chính tả:\nTriệt phá băng nhóm kướp ô tô, sử dụng \"vũ khí nóng\""

input_prompt = PROMPT_TEMPLATE.format_map(  
    {"instruction": instruction}  
)  
  
input_ids = tokenizer(input_prompt, return_tensors="pt")  
  
outputs = model.generate(  
    inputs=input_ids["input_ids"].to("cuda"),  
    attention_mask=input_ids["attention_mask"].to("cuda"),  
    do_sample=True,  
    temperature=1.0,  
    top_k=50,  
    top_p=0.9,  
    max_new_tokens=1024,  
    eos_token_id=tokenizer.eos_token_id,  
    pad_token_id=tokenizer.pad_token_id  
)  
  
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]  
response = response.split("### Trả lời:")[1]
