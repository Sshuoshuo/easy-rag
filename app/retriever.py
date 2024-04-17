from abc import ABC
from transformers import AutoTokenizer, AutoModel
import torch
import jieba
from langchain.schema.embeddings import Embeddings
from langchain.schema import Document
from typing import List
import numpy as np
from rank_bm25 import BM25Okapi
from langchain.vectorstores import FAISS


class TextEmbedding(Embeddings, ABC):
    def __init__(self, emb_model_name_or_path, batch_size=64, max_len=512, device='cuda', **kwargs):

        super().__init__(**kwargs)
        self.model = AutoModel.from_pretrained(emb_model_name_or_path, trust_remote_code=True).half().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(emb_model_name_or_path, trust_remote_code=True)
        if 'bge' in emb_model_name_or_path:
            self.DEFAULT_QUERY_BGE_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："
        else:
            self.DEFAULT_QUERY_BGE_INSTRUCTION_ZH = ""
        self.emb_model_name_or_path = emb_model_name_or_path
        self.device = device
        self.batch_size = batch_size
        self.max_len = max_len
        print("successful load embedding model")

    def compute_kernel_bias(self, vecs, n_components=384):
        """
            bertWhitening: https://spaces.ac.cn/archives/8069
            计算kernel和bias
            vecs.shape = [num_samples, embedding_size]，
            最后的变换：y = (x + bias).dot(kernel)
        """
        mu = vecs.mean(axis=0, keepdims=True)
        cov = np.cov(vecs.T)
        u, s, vh = np.linalg.svd(cov)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))
        return W[:, :n_components], -mu

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
            Compute corpus embeddings using a HuggingFace transformer model.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """
        num_texts = len(texts)
        texts = [t.replace("\n", " ") for t in texts]
        sentence_embeddings = []

        for start in range(0, num_texts, self.batch_size):
            end = min(start + self.batch_size, num_texts)
            batch_texts = texts[start:end]
            encoded_input = self.tokenizer(batch_texts, max_length=512, padding=True, truncation=True,
                                           return_tensors='pt').to(self.device)

            with torch.no_grad():

                model_output = self.model(**encoded_input)
                # Perform pooling. In this case, cls pooling.
                if 'gte' in self.emb_model_name_or_path:
                    batch_embeddings = model_output.last_hidden_state[:, 0]
                else:
                    batch_embeddings = model_output[0][:, 0]

                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                sentence_embeddings.extend(batch_embeddings.tolist())

        # sentence_embeddings = np.array(sentence_embeddings)
        # self.W, self.mu = self.compute_kernel_bias(sentence_embeddings)
        # sentence_embeddings = (sentence_embeddings+self.mu) @ self.W
        # self.W, self.mu = torch.from_numpy(self.W).cuda(), torch.from_numpy(self.mu).cuda()
        return sentence_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
            Compute query embeddings using a HuggingFace transformer model.
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        if 'bge' in self.emb_model_name_or_path:
            encoded_input = self.tokenizer([self.DEFAULT_QUERY_BGE_INSTRUCTION_ZH + text], padding=True,
                                           truncation=True, return_tensors='pt').to(self.device)
        else:
            encoded_input = self.tokenizer([text], padding=True,
                                           truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        # sentence_embeddings = (sentence_embeddings + self.mu) @ self.W
        return sentence_embeddings[0].tolist()


class Retriever:
    def __init__(self, emb_model_name_or_path=None, corpus=None, device='cuda', lan='zh'):
        self.device = device
        self.langchain_corpus = [Document(page_content=t) for t in corpus]
        self.corpus = corpus
        self.lan = lan
        if lan=='zh':
            tokenized_documents = [jieba.lcut(doc) for doc in corpus]
        else:
            tokenized_documents = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_documents)

        self.emb_model = TextEmbedding(emb_model_name_or_path=emb_model_name_or_path)
        self.db = FAISS.from_documents(self.langchain_corpus, self.emb_model)

    def bm25_retrieval(self, query, n=10):

        # 此处中文使用jieba分词
        query = jieba.lcut(query)  # 分词
        res = self.bm25.get_top_n(query, self.corpus, n=n)
        return res

    def emb_retrieval(self, query, k=10):

        search_docs = self.db.similarity_search(query, k=k)
        res = [doc.page_content for doc in search_docs]
        return res

    def retrieval(self, query, methods=None):
        if methods is None:
            methods = ['bm25', 'emb']
        search_res = list()
        for method in methods:
            if method == 'bm25':
                bm25_res = self.bm25_retrieval(query)
                for item in bm25_res:
                    if item not in search_res:
                        search_res.append(item)
            elif method == 'emb':
                emb_res = self.emb_retrieval(query)
                for item in emb_res:
                    if item not in search_res:
                        search_res.append(item)
        return search_res
