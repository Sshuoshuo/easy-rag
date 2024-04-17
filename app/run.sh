#!/bin/bash

# 这里可以放入代码运行命令
echo "program start..."
CUDA_VISIBLE_DEVICES=3 python app/run.py \
      --llm_model_name_or_path /data2/sss/LLM-hf/Qwen-7B-Chat-hf/ \
      --emb_model_name_or_path /data2/sss/LLM-hf/bge-large-zh-v1.5/ \
      --rerank_model_name_or_path /data2/sss/LLM-hf/bge-reranker-large-hf/ \
      --corpus_path app/dataset/tianchi/初赛训练数据集.pdf \
      --test_query_path app/dataset/tianchi/测试问题.json