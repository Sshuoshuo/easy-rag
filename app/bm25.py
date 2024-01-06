import jieba
from rank_bm25 import BM25Okapi
# from langchain.vectorstores import FAISS



class BM25Model:
    def __init__(self, data_list):
        tokenized_documents = [jieba.lcut(doc) for doc in data_list]
        self.bm25 = BM25Okapi(tokenized_documents)
        self.data_list = data_list
        # self.bm25_retriever = BM25Retriever.from_documents(data_list)
        # self.bm25_retriever.k = k

    def bm25_similarity(self, query, k = 10):
        query = jieba.lcut(query)  # 分词
        res = self.bm25.get_top_n(query, self.data_list, n=k)
        return res


if __name__ == '__main__':

    data_list = ["小丁的文章不好看", "我讨厌小丁的创作内容", "我非常喜欢小丁写的文章"]
    BM25 = BM25Model(data_list)
    query = "我喜欢小丁写的文章我讨厌小丁的创作内容"
    print(BM25.bm25_similarity(query, k = 2))

    query = "我讨厌小丁的创作内容"
    print(BM25.bm25_similarity(query, k=2))