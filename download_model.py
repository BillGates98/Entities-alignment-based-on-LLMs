from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizers = []
models = []

models = {
    "mistral": ["mistralai/Mistral-7B-v0.1", "./dir_models/mistral"],
    "qwen": ["Qwen/Qwen1.5-7B-Chat", "./dir_models/qwen"],
    "bloom": ["bigscience/bloom-7b1", "./dir_models/bloom"],
    "gemma": ["google/gemma-2b-it", "./dir_models/gemma"]
}

for m in models:
    print('Model ', m, ' is downloading ...')
    tokenizer = AutoTokenizer.from_pretrained(models[m][0])
    model = AutoModelForCausalLM.from_pretrained(models[m][0])
    tokenizer.save_pretrained(models[m][1])
    model.save_pretrained(models[m][1])
    print('Model ', m, ' is 100% downloaded ...')
