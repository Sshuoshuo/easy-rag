FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:2.0.0-py3.9.12-cuda11.8.0-u22.04
#FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:2.1-cuda12.2-devel-ubuntu22.04
#FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
# 避免交互式提示
#ENV DEBIAN_FRONTEND=noninteractive

# 如有安装其他软件的需求
#RUN apt-get update && apt-get install curl
#ENV LD_LIBRARY_PATH=/usr/local/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
# 如果安装其他python包的情况
RUN pip3 install transformers jieba vllm peft sentencepiece protobuf ninja langchain rank_bm25 spacy PyPDF2 tqdm transformers_stream_generator einops tiktoken faiss-gpu packaging --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

#RUN apt install -y vim
#RUN apt update
#RUN apt install -y git
#RUN apt install -y gcc-10 g++-10
#RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 60 --slave /usr/bin/g++ g++ /usr/bin/g++-10

# 复制代码到镜像仓库
COPY app /app
#COPY models/Qwen-7B-Chat-hf /app/Qwen-7B-Chat-hf
COPY models/gte-large-zh /app/gte-large-zh
COPY models/bge-large-zh /app/bge-large-zh
COPY models/bge-reranker-large-hf /app/bge-reranker-large-hf
#COPY models/m3e-large-hf /app/m3e-large-hf
# 指定工作目录
WORKDIR /app
#RUN pip3 install zh_core_web_sm-3.7.0-py3-none-any.whl --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

#RUN cd flash-attention-main/ && pip install . && pip install csrc/layer_norm && pip install csrc/rotary && cd ..

# 容器启动运行命令
CMD ["bash", "run.sh"]
