# easy-rag

本项目由天池rag比赛项目解决方案修改而来，目标是帮助rag入门者理解整个rag的具体流程。

最终成绩复赛第四名，链接如下。

[2023全球智能汽车AI挑战赛——赛道一：AI大模型检索问答_算法大赛_天池大赛-阿里云天池的赛制 (aliyun.com)](https://tianchi.aliyun.com/competition/entrance/532154)

# 文件目录

```
. 
├── app # 代码文件
│   ├── run.py # 主函数入口，演示demo数据集
│   ├── retriever.py  # 检索器，包含向量检索与BM25关键词检索
│   ├── reranker.py # 重排器，精排
│   ├── llm_infer.py # 大模型推理，包括一些示例模版
│   ├── read_corpus.py # 语料库/知识库读取
│   ├── run.py # 主函数入口，演示demo数据集
│   ├── run.sh # 运行脚本
├── dataset # 示例数据集
│   ├── tianchi # 示例数据，以PDF形式给出
│   │   ├── 测试问题.json # 测试问题
│   │   └── 初赛训练数据集.pdf # 外部知识
├── requirement.txt # 示例环境
└── README.md # 本文件
```

# 运行

## 国内用户下载模型

1、Qwen-7B-Chat/Qwen1.5-7B-Chat

下载地址：[Qwen/Qwen1.5-7B-Chat · Hugging Face](https://huggingface.co/Qwen/Qwen1.5-7B-Chat)

2、bge-large-zh

下载地址：[BAAI/bge-large-zh · Hugging Face](https://huggingface.co/BAAI/bge-large-zh)

3、bge-reranker-large

下载地址：[BAAI/bge-reranker-large · Hugging Face](https://huggingface.co/BAAI/bge-reranker-large)

## 脚本

```bash
bash app/run.sh
```

## 替换为私有数据

1、完成数据读取，即read_corpus.py中的extract_my_file()函数，返回格式为List[str]

2、测试问题读取，直接在run.py中修改

3、只使用BM25或者向量检索，主函数中“--retrieval_methods”参数修改为['bm25']或['emb']

4、更换其他大模型，修改llm_infer.py中my_llm_infer()函数，以及__init__中的模型加载

5、其他未使用的模板包括部分优化方案：

​		（1）将抽取后的文档使用LLM重新整理，使得杂乱知识库规整

​		（2）一次给LLM一个检索到的文档，不断优化生成的答案

​		（3）先试用LLM直接生成答案，然后将问题和这个生成的答案拼接，共同完成检索，提升检索效果

# TODO

1、完善多类型知识库读取，目前只展示了一个PDF示例

2、加入检索器与重排器微调，提高效果

3、加入检索、重排、RAG评价指标，展示各个模型与流程的效果

4、后续demo数据集计划使用[Multi-CPR: 大规模段落检索多领域中文数据集_数据集-阿里云天池 (aliyun.com)](https://tianchi.aliyun.com/dataset/132745)

5、优化方案
