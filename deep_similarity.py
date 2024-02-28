
import openai
from tqdm import tqdm


class DeepSimilarity:

    def __init__(self, model_name='', model=None):
        self.model_name = model_name
        self.tokenizer, self.model = model
        print('model name : ', model_name)

    def mistral(self, query=""):
        input_ids = self.tokenizer(query, return_tensors="pt")
        outputs = self.model.generate(**input_ids, max_length=200)
        output = self.tokenizer.decode(outputs[0])
        if 'yes' in output.lower():
            return 'yes'
        return 'no'

    def qwen(self, query=''):
        input_ids = self.tokenizer(query, return_tensors="pt")
        outputs = self.model.generate(**input_ids, max_length=200)
        output = self.tokenizer.decode(outputs[0])
        if 'yes' in output.lower():
            return 'yes'
        return 'no'

    def gpt(self, query=''):
        input_ids = self.tokenizer(query, return_tensors="pt")
        outputs = self.model.generate(**input_ids, max_length=200)
        output = self.tokenizer.decode(outputs[0])
        if 'yes' in output.lower():
            return 'yes'
        return 'no'

    def bloom(self, query=''):
        input_ids = self.tokenizer(query, return_tensors="pt")
        outputs = self.model.generate(**input_ids, max_length=200)
        output = self.tokenizer.decode(outputs[0])
        if 'yes' in output.lower():
            return 'yes'
        return 'no'

    def run(self, query=''):
        if self.model_name == 'mistral':
            return self.mistral(query=query)
        elif self.model_name == 'qwen':
            return self.qwen(query=query)
        elif self.model_name == 'gpt':
            return self.gpt(query=query)
        elif self.model_name == 'bloom':
            return self.bloom(query=query)
        return ''
