import torch
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def build_simple_template():
    prompt_template = "你是一个准确和可靠的人工智能助手，请准确回答下面的用户问题。\n" \
                        "用户问题：\n" \
                        "{}\n"
    return prompt_template


def build_template():
    prompt_template = "你是一个准确和可靠的人工智能助手，能够借助外部文档回答用户问题，请注意外部文档可能存在噪声事实性错误。" \
                      "如果文档中的信息包含了正确答案，你将进行准确的回答。"\
                      "如果文档中的信息不包含答案，你将生成“文档信息不足，因此我无法基于提供的文档回答该问题。”。" \
                      "如果部分文档中存在与事实不一致的错误，请先生成“提供文档的文档存在事实性错误。”，并生成正确答案。" \
                      "下面给定你相关外部文档，根据文档来回答用户问题。" \
                      "以下是外部文档：\n---" \
                        "{}\n" \
                        "用户问题：\n---" \
                        "{}\n"
    return prompt_template


def build_summary_template():
    prompt_template = "请你将给定的杂乱文本重新整理，使其不丢失任何信息且有较强的可读性，同时要求不丢失关键词。\n" \
                      "以下是杂乱文本：\n---" \
                      "{}\n" \

    return prompt_template


def build_repair_template():

    prompt_template = "你是一个准确和可靠的人工智能助手。" \
                      "请你基于以下材料调整优化用户问题的答案，要求答案尽可能的清晰准确，并且包含正确的关键词。" \
                      "如果没有必要调整则将原答案重复即可。\n" \
                      "以下是材料：\n---" \
                      "{}\n" \
                      "用户问题：\n---" \
                        "{}\n" \
                        "原答案：\n---" \
                        "{}\n"

    return prompt_template


class LLMPredictor(object):
    def __init__(self, llm_model_name_or_path, device="cuda", **kwargs):

        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name_or_path,
            use_fast=True,
            padding_side="right",
            trust_remote_code=True,
        )

        config = AutoConfig.from_pretrained(llm_model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(llm_model_name_or_path, config=config, device_map='cuda',
                                                     torch_dtype=torch.float16, trust_remote_code=True)

        self.max_token = 4096
        self.simple_template = build_simple_template()
        self.prompt_template = build_template()
        self.repair_template = build_repair_template()
        self.summary_template = build_summary_template()
        self.kwargs = kwargs
        self.device = device
        self.model_path = llm_model_name_or_path
        print('successful load LLM: ', llm_model_name_or_path)

    def qwen_infer(self, prompt, device='cuda'):
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

    def predict(self, context, query):

        input_ids = self.tokenizer(context, return_tensors="pt", add_special_tokens=False).input_ids
        if len(input_ids) > self.max_token:
            context = self.tokenizer.decode(input_ids[:self.max_token-1])
            warnings.warn("texts have been truncted")
        context = self.prompt_template.format(context, query)
        if 'qwen' in self.model_path.lower():
            response = self.qwen_infer(context)
        else:
            response = self.my_llm_infer(context)
        return response

    def get_prompt(self, context, query):

        content = self.prompt_template.format(context, query)

        return content

    def repair_answer(self, context, query, origin_answer):

        input_ids = self.tokenizer(context, return_tensors="pt", add_special_tokens=False).input_ids
        if len(input_ids) > self.max_token:
            context = self.tokenizer.decode(input_ids[:self.max_token-1])
            warnings.warn("texts have been truncted")
        context = self.repair_template.format(context, query, origin_answer)
        if 'qwen' in self.model_path.lower():
            response = self.qwen_infer(context)
        else:
            response = self.my_llm_infer(context)
        return response

    def simple_predict(self, query):

        prompt = self.simple_template.format(query)
        if 'qwen' in self.model_path.lower():
            response = self.qwen_infer(prompt)
        else:
            response = self.my_llm_infer(prompt)
        return response

    def construct_search_docs(self, context):

        context = self.summary_template.format(context)
        if 'qwen' in self.model_path.lower():
            response = self.qwen_infer(context)
        else:
            response = self.my_llm_infer(context)
        return response

    def my_llm_infer(self, prompt, device='cuda'):

        raise NotImplementedError