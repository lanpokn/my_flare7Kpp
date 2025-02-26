# 使用 CUDA 12.4 作为基础镜像
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 更新 apt，并安装基本依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y libgl1-mesa-glx && apt-get install -y libglib2.0-0

# 确保 Python3 使用 `python` 命令
RUN ln -s /usr/bin/python3 /usr/bin/python

# 复制项目文件（包括 setup.py）
COPY . /app

# 安装 Python 依赖
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# 运行 setup.py develop
RUN python setup.py develop

# 设定容器启动时的默认命令
CMD ["/bin/bash"]