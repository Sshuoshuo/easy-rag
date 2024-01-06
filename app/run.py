import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from LLM import LLMPredictor
from embeddings import BGEpeftEmbedding
from langchain import FAISS
from pdfparser import extract_page_text
from bm25 import BM25Model
import torch
import jieba
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids
# torch.cuda.set_per_process_memory_fraction(0.93)
def rerank(docs, query, rerank_tokenizer, rerank_model, k=5):
    docs_ = []
    for item in docs:
        if isinstance(item, str):
            docs_.append(item)
        else:
            docs_.append(item.page_content)
    docs = list(set(docs_))
    pairs = []
    for d in docs:
        pairs.append([query, d])
    with torch.no_grad():
        inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to('cuda')
        # for key in inputs:
        #     inputs[key] = inputs[key].to('cuda')
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float().cpu().tolist()
    docs = [(docs[i], scores[i]) for i in range(len(docs))]
    docs = sorted(docs, key = lambda x: x[1], reverse = True)
    docs_ = []
    for item in docs:
        # if item[1]>0:
        #     docs_.append(item[0])
        docs_.append(item[0])
    return docs_[:k]

def create_json_line(text):
    line_dict = {"text": text}
    json_line = json.dumps(line_dict)
    return json_line

from vllm import SamplingParams
sampling_params = SamplingParams(temperature=1.0, top_p=0.5, max_tokens=512) #temperature=1.0, top_p=0.5, max_tokens=512
def infer_by_batch(all_raw_text, llm, system="你是一个汽车驾驶安全员,精通有关汽车驾驶、维修和保养的相关知识。"):
    batch_raw_text = []
    for q in all_raw_text:
        raw_text, _ = make_context(
            llm.tokenizer,
            q,
            system=system,
            max_window_size=6144,
            chat_format='chatml',
        )
        batch_raw_text.append(raw_text)
    res = llm.model.generate(batch_raw_text, sampling_params, use_tqdm = False)
    res = [output.outputs[0].text.replace('<|im_end|>', '').replace('\n', '') for output in res]
    return res

def post_process(answer):
    if '抱歉' in answer or '无法回答' in answer or '无答案' in answer:
        return "无答案"
    return answer

def main():
    submit = True
    batch_size = 4
    num_input_docs = 4
    model = "../models/Qwen-7B-Chat-hf" if not submit else "/tcdata/qwen/Qwen-7B-Chat"
    embedding_path = "../models/gte-large-zh" if not submit else "gte-large-zh"
    embedding_path2 = "../models/bge-large-zh" if not submit else "bge-large-zh"
    reranker_model_path = "../models/bge-reranker-large-hf" if not submit else "bge-reranker-large-hf"

    llm = LLMPredictor(model_path=model, is_chatglm=False, device='cuda:0') # , temperature = 0.5; temperature=1.0, top_p=0.5
    # llm.model.config.use_flash_attn = True

    rerank_tokenizer = AutoTokenizer.from_pretrained(reranker_model_path)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_path)
    rerank_model.eval()
    rerank_model.half()
    rerank_model.cuda()

    filepath = "初赛训练数据集.pdf" if not submit else "/tcdata/trainning_data.pdf"
    docs = extract_page_text(filepath=filepath, max_len=300, overlap_len=100) + extract_page_text(filepath=filepath, max_len=500, overlap_len=200)

    corpus = [item.page_content for item in docs]

    # embedding database
    embedding_model = BGEpeftEmbedding(model_path=embedding_path)
    db = FAISS.from_documents(docs, embedding_model)
    embedding_model2 = BGEpeftEmbedding(model_path=embedding_path2)
    db2 = FAISS.from_documents(docs, embedding_model2)

    BM25 = BM25Model(corpus)

    result_list = []
    test_file = "测试问题.json" if not submit else "/tcdata/test_question.json"
    with open(test_file, 'r', encoding='utf-8') as f:
        result = json.load(f)

    prompts1, prompts2, prompts3 = [], [], []
    all_prompts = []
    all_prompts1 = []
    ress1, ress2, ress3 = [], [], []
    for i, line in tqdm(enumerate(result)):
        # bm25 召回
        search_docs = BM25.bm25_similarity(line['question']*3, 10)
        # bge 召回
        search_docs2 = db2.similarity_search(line['question']*3, k=10)
        # gte 召回
        search_docs3 = db.similarity_search(line['question']*3, k=10)
        # rerank
        search_docs4 = rerank(search_docs + search_docs2 + search_docs3, line['question'], rerank_tokenizer, rerank_model, k=num_input_docs)

        prompt1 = llm.get_prompt("\n".join(search_docs4[::-1]), line['question'], bm25=True)
        prompt2 = llm.get_prompt("\n".join(search_docs[:num_input_docs][::-1]), line['question'], bm25=True)
        # prompt3 = llm.get_prompt("\n".join(search_docs5[::-1]), line['question'], bm25=True)
        prompts1.append(prompt1)
        prompts2.append(prompt2)
        # prompts3.append(prompt3)
        all_prompts1.append(search_docs4[0]+'\n'+search_docs[1]+'\n'+search_docs[2])
        all_prompts.append(search_docs[0]+'\n'+search_docs[1]+'\n'+search_docs[2])

        if len(prompts1)==batch_size:
            ress1.extend(infer_by_batch(prompts1, llm))
            prompts1 = []
            ress2.extend(infer_by_batch(prompts2, llm))
            prompts2 = []
            # ress3.extend(infer_by_batch(prompts3, llm))
            # prompts3 = []

    if len(prompts1)>0:
        ress1.extend(infer_by_batch(prompts1, llm))
        ress2.extend(infer_by_batch(prompts2, llm))
        # ress3.extend(infer_by_batch(prompts3, llm))


    for i, line in enumerate(result):
        res1 = post_process(ress1[i])
        res2 = post_process(ress2[i])

        question_keywords = jieba.lcut(line['question'])
        no_answer = True
        context = all_prompts[i]
        for kw in question_keywords:
            if kw in context:
                no_answer = False
                break
        if no_answer:
            res3 = '无答案'
        else:
            res3 = res2 + '\n参考：' + context
        res3 = post_process(res3)

        line['answer_1'] = res1
        line['answer_2'] = res2
        line['answer_3'] = res3
        result_list.append(line)


    res_file_path = 'res.json' if not submit else "/app/result.json"
    with open(res_file_path, 'w', encoding='utf-8') as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()


