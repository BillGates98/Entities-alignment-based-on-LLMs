from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
# self.models = {
#             "mistral": ["mistralai/Mistral-7B-v0.1", "./dir_models/mistral/Mistral-7B-v0.1"],
#             "qwen": ["Qwen/Qwen1.5-7B-Chat", "./dir_models/qwen/Qwen1.5-7B-Chat"],
#             "bloom": ["bigscience/bloom-7b1", "./dir_models/bloom/bloom-7b1"],
#             "gpt": ["openai-community/gpt2", "./dir_models/gpt/gpt2"]
#         }


class LLM:

    def __init__(self, model_name=None):
        self.model_name = model_name
        self.models = {
            "mistral": ["mistralai/Mistral-7B-v0.1", "./dir_models/mistral"],
            "qwen": ["Qwen/Qwen1.5-7B-Chat", "./dir_models/qwen"],
            "bloom": ["bigscience/bloom-7b1", "./dir_models/bloom"],
            "gpt": ["openai-community/gpt2", "./dir_models/gpt"]
        }

    def mistral(self):
        tokenizer = AutoTokenizer.from_pretrained(self.models['mistral'][1])
        model = AutoModelForCausalLM.from_pretrained(
            self.models['mistral'][1], device_map="auto", trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained(
            self.models['mistral'][1])
        return (tokenizer, model)

    def qwen(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.models['qwen'][1], device_map="auto", trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.models['qwen'][1])
        model.generation_config = GenerationConfig.from_pretrained(
            self.models['qwen'][1])
        return (tokenizer, model)

    def bloom(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.models['bloom'][1], device_map="auto", trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.models['bloom'][1])
        model.generation_config = GenerationConfig.from_pretrained(
            self.models['bloom'][1])
        return (tokenizer, model)

    def gpt(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.models['gpt'][1], device_map="auto", trust_remote_code=True).eval()
        model = AutoModelForCausalLM.from_pretrained(self.models['gpt'][1])
        model.generation_config = GenerationConfig.from_pretrained(
            self.models['gpt'][1])
        return (tokenizer, model)

    def load(self):
        if self.model_name == 'mistral':
            return self.mistral()
        elif self.model_name == 'qwen':
            return self.qwen()
        elif self.model_name == 'gpt':
            return self.gpt()
        elif self.model_name == 'bloom':
            return self.bloom()
        return (None, None)
