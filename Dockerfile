# 使用 PaddlePaddle 官方镜像，包含 CUDA 11.8 和 CUDNN 8.6
# 即使宿主机是 CUDA 12.8，Docker 内部可以使用 11.8
FROM registry.baidubce.com/paddlepaddle/paddle:2.6.1-gpu-cuda11.7-cudnn8.4-trt8.4

# 设置工作目录
WORKDIR /app

# 设置时区和语言环境
ENV TZ=Asia/Shanghai
ENV LANG=C.UTF-8

# 安装系统依赖 (OpenCV 和 PyMuPDF 需要的库)
#以此提高构建速度，更换为阿里源(可选)
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
RUN pip config set global.trusted-host mirrors.aliyun.com
RUN pip install --no-cache-dir -r requirements.txt

# 预下载 PaddleOCR 模型 (可选，防止启动时下载超时)
# 这一步不是必须的，但生产环境建议这么做。
# 如果不加这一行，第一次请求时会自动下载模型到 /root/.paddleocr
RUN python3 -c "from paddleocr import PaddleOCR; PaddleOCR(lang='ch', use_gpu=False, show_log=False)"