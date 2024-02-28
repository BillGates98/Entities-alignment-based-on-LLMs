
import openai
from tqdm import tqdm


class DeepSimilarity:

    def __init__(self, model_name='', model=None):
        self.model_name = model_name
        self.tokenizer, self.model = model
        print('model name : ', model_name)

    def llama(self, query=""):
        question = f"{query}"
        system_message = 'you are a helpful assistant!'
        inputs = {'text': question,
                  'system': system_message, 'max_length': 4}
        response = self.model.chat(inputs, self.tokenizer)
        return response['response']

    def qwen(self, query=''):
        question = f"{query}"
        response, _ = self.model.chat(self.tokenizer, question, history=None)
        response_cleaned = response.replace("\n", '')
        return response_cleaned

    def gpt(self, query=''):
        try:
            completion = openai.ChatCompletion.create(model="gpt-4",
                                                      messages=[{"role": "assistant", "content": query}])
            response = str(
                completion.choices[0].message["content"]).replace("\n", "")
            return response
        except Exception as e:
            return 'no'

    def bloom(self, query=''):
        inputs = self.tokenizer(query, return_tensors="pt")
        result_length = 4
        output = (self.tokenizer.decode(self.model.generate(inputs["input_ids"],
                                                            max_length=result_length,
                                                            num_beams=2,
                                                            no_repeat_ngram_size=2,
                                                            early_stopping=True
                                                            )[0]))
        if 'yes' in output.lower():
            return 'yes'
        return 'no'

    def gemma(self, query=''):
        input_ids = self.tokenizer(query, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**input_ids, max_length=200)
        output = self.tokenizer.decode(outputs[0])
        if 'yes' in output.lower():
            return 'yes'
        return 'no'

    def run(self, query=''):
        if self.model_name == 'llama':
            return self.llama(query=query)
        elif self.model_name == 'qwen':
            return self.qwen(query=query)
        elif self.model_name == 'gpt':
            return self.gpt(query=query)
        elif self.model_name == 'bloom':
            return self.bloom(query=query)
        return ''
