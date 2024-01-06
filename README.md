# AI大模型检索问答

比赛项目解决方案，最终成绩复赛第四名，链接如下。

[2023全球智能汽车AI挑战赛——赛道一：AI大模型检索问答_算法大赛_天池大赛-阿里云天池的赛制 (aliyun.com)](https://tianchi.aliyun.com/competition/entrance/532154)

感谢@[poisonwine](https://github.com/poisonwine)的Baseline，[poisonwine/Tianchi-LLM-retrieval: 2023全球智能汽车AI挑战赛——赛道一：AI大模型检索问答， 75+ baseline (github.com)](https://github.com/poisonwine/Tianchi-LLM-retrieval?spm=a2c22.21852664.0.0.26c77aadYH1Uwe)

# 文件目录

```
. 
├── app # 代码文件
│   ├── bm25.py   # BM25召回模型，依赖jieba分词与rank_bm25库中的BM250kapi模型
│   ├── embeddings.py  # 向量召回嵌入模型，包括对bge和其他模型的嵌入方法，bge使用指令的方式需加前缀
│   ├── LLM.py # 大模型推理，主要包括vllm的初始化，prompt模板构建，chat推理
│   ├── pdfparser.py # PDF解析与分块，采用简单的正则去除页码信息，采用滑动窗口切块
│   ├── qwen_generation_utils.py # qwen工具，直接从qwen开源代码中复制来，主要为了构建qwen的batch推理输入形式
│   ├── run.py # 主函数入口，流程后续详细介绍
│   ├── run.sh # 运行脚本，python3 run.py
│   ├── 测试问题.json # a榜测试问题，调试代码用
│   └── 初赛训练数据集.pdf # a榜训练数据，调试代码用
├── models # 模型文件，本地调试时将模型全放在此文件夹中
├── Dockerfile # Dockerfile，用于创建镜像
└── README.md # 本文件
```

# 主要处理流程

## 使用模型及其对应下载路径

1、Qwen-7B-Chat

使用官方线上模型，下载地址略；

2、gte-large-zh

下载地址：[thenlper/gte-large-zh · Hugging Face](https://huggingface.co/thenlper/gte-large-zh)

3、bge-large-zh

下载地址：[BAAI/bge-large-zh · Hugging Face](https://huggingface.co/BAAI/bge-large-zh)

4、bge-reranker-large

下载地址：[BAAI/bge-reranker-large · Hugging Face](https://huggingface.co/BAAI/bge-reranker-large)

## 主要流程

### step1 加载模型

加载LLM、加载embedding模型、加载reranker模型，且都放到GPU上

### step2 解析PDF

extract_page_text函数为解析PDF函数，将PDF解析两次，一次是长度300的块，前后重叠100长度；另一次是长度500的块，前后重叠200；两次合并起来构成总的解析数据集，用于后续召回；

### step3 知识库构建

主要包括向量知识库构建、BM25知识库构建

### step4 相关知识召回与生成

多路召回与排序。多路召回包括bm25召回、bge召回、gte召回，然后使用bge-reranker进行精排，选取得分最高的前4个作为输入到llm的上下文。据此生成答案1。

考虑到此题有关键词得分，因此bm25召回对于关键词有较大的作用，因此使用bm25召回的前4个作为输入到llm的上下文进行推理。据此生成答案2。

由于有些问题无答案，因此对于答案3加一层关键词判断，使用jieba对question进行分词，若所有的词均未在检索的文档中出现，则将答案3视为无答案，否则将答案3赋为答案2，并且为了进一步提高答案2的可靠性，在答案2的基础上给与了依据文档。

### step5 后处理

对于大模型拒绝回答的通常会包含“抱歉”、“无法回答”等词，后处理会将这些回答替换为标准的“无答案”

## 关键点

1、PDF解析多次，进行召回时自适应召回不同块的大小，尽可能减少输入到llm的噪声。

2、多路召回+精排的流程，而非简单的召回

3、进行召回时将question复制三倍，使得question长度与召回文档长度不会相差太多，有利于提升召回效果

4、向量编码尝试了bert-whitening

5、使用关键词判断是否无答案

6、推理加速：使用vllm并且根据Qwen官方脚本完成batch推理，推理速度有较大提升
