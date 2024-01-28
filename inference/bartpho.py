import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer




#BARTpho-word
word_tokenizer = AutoTokenizer.from_pretrained("Bachhoang/BARTpho-vietnamese-legal")
bartpho_word = AutoModelForSeq2SeqLM.from_pretrained("Bachhoang/BARTpho-vietnamese-legal")
TXT = '##### Câu hỏi: Nội dung của Điều 2 788/QĐ-BTNMT là gì? ### Trả lời:'  
input_ids = word_tokenizer(TXT, return_tensors='pt')['input_ids']
features = bartpho_word(input_ids)
