# Finetuning vietnamese three NLP model (bert, gpt, llama2) for legal action
Do following steps:
```
git clone https://github.com/bachhoang0606/BARTPho-finetuning.git
cd BARTPho-finetuning
```
### BARTPho ([Hugging Face](https://huggingface.co/docs/transformers/model_doc/bartpho#usage-example))
```
conda create -n BARTpho_finetune python=3.9
conda activate BARTpho_finetune
./run-bart.sh
```
### vietnamese-llama2-7b-120GB ([Hugging Face](https://huggingface.co/bkai-foundation-models/vietnamese-llama2-7b-120GB))
```
conda create -n BARTpho_finetune python=3.9
conda activate BARTpho_finetune
./run-llama2.sh
```
### PhoGPT ([Hugging Face](https://huggingface.co/vinai/PhoGPT-7B5))
```
conda create -n BARTpho_finetune python=3.9
conda activate BARTpho_finetune
./run-gpt.sh
```
Note: if problem with `./run.sh`, run ```chmod +x run.sh```