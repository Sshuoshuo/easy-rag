from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, GenerationConfig
# from transformers import BitsAndBytesConfig
import torch
from vllm import LLM
# from peft import PeftModel
# from langchain.schema.embeddings import Embeddings
# from typing import List
# import bitsandbytes as bnb
import warnings

def build_simple_template():
    prompt_template = "你是一个汽车驾驶安全员,精通有关汽车驾驶、维修和保养的相关知识。请你使用自己的知识回答用户问题。回答要清晰准确，包含正确关键词。如果所给材料与用户问题无关，请回答：无答案\n" \
                        "用户问题：\n" \
                        "{}\n"
    return prompt_template

def build_template():
    prompt_template = "请你基于以下材料回答用户问题。回答要清晰准确，包含正确关键词。不要胡编乱造。如果所给材料与用户问题无关，只输出：无答案。\n" \
                      "以下是材料：\n---" \
                        "{}\n" \
                        "用户问题：\n" \
                        "{}\n" \
                        "务必注意，如果所给材料无法回答用户问题，只输出无答案，不要自己回答。"
    return prompt_template

def build_summary_template():
    prompt_template = "请你将给定的杂乱文本重新整理，使其不丢失任何信息且有较强的可读性，同时要求不丢失关键词。\n" \
                      "以下是杂乱文本：\n---" \
                      "{}\n" \

    return prompt_template

def build_repair_template():

    prompt_template = "你是一个汽车驾驶安全员,精通有关汽车驾驶、维修和保养的相关知识。请你基于以下汽车手册材料调整优化用户问题的答案，要求答案尽可能的清晰准确，并且包含正确的关键词。如果没有必要调整则将原答案重复即可。\n" \
                      "以下是材料：\n---" \
                      "{}\n" \
                      "用户问题：\n" \
                        "{}\n" \
                        "原答案：\n" \
                        "{}\n"

    return prompt_template


class LLMPredictor(object):
    def __init__(self, model_path, adapter_path=None, is_chatglm=False, device="cuda", **kwargs):

        # self.tokenizer = AutoTokenizer.from_pretrained(model_path,
        #                                                 pad_token='<|extra_0|>',
        #                                                 eos_token='<|endoftext|>',
        #                                                 padding_side='left',
        #                                                 trust_remote_code=True
        #                                             )

        self.model = LLM(model=model_path, trust_remote_code=True, dtype = 'bfloat16', gpu_memory_utilization = 0.85)
        self.tokenizer = self.model.get_tokenizer()
        self.tokenizer.pad_token='<|extra_0|>'
        self.tokenizer.eos_token='<|endoftext|>'
        self.tokenizer.padding_side='left'

        self.max_token = 4096
        self.simple_template = build_simple_template()
        self.prompt_template = build_template()
        self.repair_template = build_repair_template()
        self.summary_template = build_summary_template()
        self.kwargs = kwargs
        self.device = torch.device(device)
        print('successful load LLM', model_path)

        self.model_path = model_path


    def predict(self, context, query, bm25 = False, is_yi = False):

        if bm25:
            content = context
        else:
            content = "\n".join(doc.page_content for doc in context)
        if "deepseek" in self.model_path or is_yi:
            content = self.prompt_template.format(content, query)
            messages = [
                {"role": "user", "content": content}
            ]
            input_tensor = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", tokenize=True)
            outputs = self.model.generate(input_tensor.to(self.model.device), max_new_tokens=500, **self.kwargs)
            result = self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
            return result
        input_ids = self.tokenizer(content, return_tensors="pt", add_special_tokens=False).input_ids
        if len(input_ids) > self.max_token:
            content = self.tokenizer.decode(input_ids[:self.max_token-1])
            warnings.warn("texts have been truncted")
        content = self.prompt_template.format(content, query)
        # print(prompt)
        response, history = self.model.chat(self.tokenizer, content, history=[], **self.kwargs)
        return response

    def get_prompt(self, context, query, bm25 = False, is_yi = False):

        if bm25:
            content = context
        else:
            content = "\n".join(doc.page_content for doc in context)
        # input_ids = self.tokenizer(content, return_tensors="pt", add_special_tokens=False).input_ids
        # if len(input_ids) > self.max_token:
        #     content = self.tokenizer.decode(input_ids[:self.max_token-1])
        #     warnings.warn("texts have been truncted")
        content = self.prompt_template.format(content, query)
        return content

    def repair_answer(self, context, query, origin_answer, bm25 = False, is_yi = False):

        if bm25:
            content = context
        else:
            content = "\n".join(doc.page_content for doc in context)
        input_ids = self.tokenizer(content, return_tensors="pt", add_special_tokens=False).input_ids
        if len(input_ids) > self.max_token:
            content = self.tokenizer.decode(input_ids[:self.max_token-1])
            warnings.warn("texts have been truncted")
        content = self.repair_template.format(content, query, origin_answer)
        response, history = self.model.chat(self.tokenizer, content, history=[], **self.kwargs)
        return response

    def simple_predict(self, query):
        prompt = self.simple_template.format(query)
        response, history = self.model.chat(self.tokenizer, prompt, history=[], **self.kwargs)
        return response

    def construct_search_docs(self, context, question,  bm25 = False):
        if bm25:
            content = context
        else:
            content = "\n".join(doc.page_content for doc in context)

        content = self.summary_template.format(content)
        return content

