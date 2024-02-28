from transformers import AutoModelForCausalLM, AutoTokenizer


class LLM:

    def __init__(self, model_name=None):
        self.model_name = model_name
        self.models = {
            "mistral": ["mistralai/Mistral-7B-v0.1", "./dir_models/mistral"],
            "qwen": ["Qwen/Qwen1.5-7B-Chat", "./dir_models/qwen"],
            "bloom": ["bigscience/bloom-7b1", "./dir_models/bloom"],
            "gemma": ["google/gemma-2b-it", "./dir_models/gemma"]
        }

    def mistral(self):
        tokenizer = AutoTokenizer.from_pretrained(self.models['mistral'][1])
        model = AutoModelForCausalLM.from_pretrained(self.models['mistral'][1])
        return (tokenizer, model)

    def qwen(self):
        model = AutoModelForCausalLM.from_pretrained(self.models['qwen'][1])
        tokenizer = AutoTokenizer.from_pretrained(self.models['qwen'][1])
        return (tokenizer, model)

    def bloom(self):
        model = AutoModelForCausalLM.from_pretrained(self.models['bloom'][1])
        tokenizer = AutoTokenizer.from_pretrained(self.models['bloom'][1])
        return (tokenizer, model)

    def gemma(self):
        tokenizer = AutoTokenizer.from_pretrained(self.models['gemma'][1])
        model = AutoModelForCausalLM.from_pretrained(self.models['gemma'][1])
        return (tokenizer, model)

    def load(self):
        if self.model_name == 'mistral':
            return self.llama()
        elif self.model_name == 'qwen':
            return self.qwen()
        elif self.model_name == 'gemma':
            return (None, None)
        elif self.model_name == 'bloom':
            return self.bloom()
        return (None, None)
