import torch
from modelscope.models.nlp.llama2 import Llama2Tokenizer
from modelscope import Model, snapshot_download

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from tqdm import tqdm

from transformers import BloomForCausalLM
from transformers import BloomForTokenClassification
from transformers import BloomTokenizerFast


class LLM:

    def __init__(self, model_name=None):
        self.model_name = model_name

    def llama(self):
        model_dir = snapshot_download("modelscope/Llama-2-7b-chat-ms", revision='v1.0.2',
                                      ignore_file_pattern=[r'.+\.bin$'])
        tokenizer = Llama2Tokenizer.from_pretrained(model_dir)
        model = Model.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map='auto')
        return (tokenizer, model)

    def qwen(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "/openbayes/input/input0", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("/openbayes/input/input0", device_map="auto",
                                                     trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained(
            "/openbayes/input/input0", trust_remote_code=True)

        return (tokenizer, model)

    def bloom(self):
        tokenizer = BloomTokenizerFast.from_pretrained(
            "bigscience/bloom-1b7", local_files_only=False)
        model = BloomForCausalLM.from_pretrained(
            "bigscience/bloom-1b7", local_files_only=False)
        return (tokenizer, model)

    def load(self):
        if self.model_name == 'llama':
            return self.llama()
        elif self.model_name == 'qwen':
            return self.qwen()
        elif self.model_name == 'gpt':
            return (None, None)
        elif self.model_name == 'bloom':
            return self.bloom()
        return (None, None)
