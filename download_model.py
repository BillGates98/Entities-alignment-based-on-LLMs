from transformers import AutoTokenizer, AutoModelForCausalLM

models = {
    "mistral": ["mistralai/Mistral-7B-v0.1", "./dir_models/mistral"],
    "qwen": ["Qwen/Qwen1.5-7B-Chat", "./dir_models/qwen"],
    "bloom": ["bigscience/bloom-7b1", "./dir_models/bloom"],
    "gpt": ["openai-community/gpt2", "./dir_models/gpt"]
}

for m in models:
    print('Model ', m, ' is downloading ...')
    tokenizer = AutoTokenizer.from_pretrained(models[m][0])
    model = AutoModelForCausalLM.from_pretrained(models[m][0])
    tokenizer.save_pretrained(models[m][1])
    model.save_pretrained(models[m][1])
    print('Model ', m, ' is 100% downloaded ...')
