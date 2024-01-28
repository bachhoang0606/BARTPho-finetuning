from utils import handle_raw_datasets
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

#BARTpho-word
word_tokenizer = AutoTokenizer.from_pretrained("Bachhoang/BARTpho-vietnamese-legal")
bartpho_word = AutoModelForSeq2SeqLM.from_pretrained("Bachhoang/BARTpho-vietnamese-legal")
max_length = 512

def inference_bartpho(text : str):

    input_ids = word_tokenizer(text, return_tensors='pt')['input_ids']
    outputs = bartpho_word.generate(input_ids, max_length=max_length, min_length=max_length)
    decoded = word_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return handle_raw_datasets(decoded, text)

if __name__ == "__main__":

    text = '##### Câu hỏi: Nội dung của Điều 11 27/2022/NQ-HĐND là gì? ### Trả lời:'  
    input_ids = word_tokenizer(text, return_tensors='pt')['input_ids']
    outputs = bartpho_word.generate(input_ids, max_length=max_length, min_length=max_length)
    decoded = word_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(handle_raw_datasets(decoded, text))
