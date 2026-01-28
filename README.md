<h1 align="center">🚀 PaddleOCR FastAPI Service</h1>
<p align="center">🚀 轻量级、高性能的 PaddleOCR + FastAPI OCR 服务</p>
<p align="center">

<div align='center'>
<a href="#" target="_blank"><img src="https://img.shields.io/badge/FastAPI-Framework-009688?logo=fastapi&style=flat-square"></a>
<a href="#" target="_blank"><img src="https://img.shields.io/badge/PaddleOCR-Project-orange?logo=paddle&style=flat-square"></a>
<a href="#" target="_blank"><img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&style=flat-square"></a>
<a href="#" target="_blank"><img src="https://img.shields.io/badge/Docker-Deploy-2496ED?logo=docker&style=flat-square"></a>
<img src="https://img.shields.io/badge/license-MIT-yellow?style=flat-square">
</div>
</p>

---

## 🌟 项目简介

该仓库提供一个最小可用的 OCR HTTP 服务，具备以下特性：

- ✨ **高性能识别**：使用 PaddleOCR 进行文字检测与识别（支持中文）。
- 📝 **多格式支持**：接收单张图片或多页 PDF（服务端自动将 PDF 每页渲染为图片）。
- 📊 **结构化输出**：返回 JSON 格式的识别结果，包含文本内容、置信度和坐标框。
- 🐳 **容器化部署**：支持 Docker & Docker Compose，开箱即用。

## 📂 文件结构

- `main.py` - FastAPI 应用，主要逻辑与路由实现。
- `requirements.txt` - Python 依赖。
- `Dockerfile`, `docker-compose.yml` - 容器化部署配置。

---

## ⚙️ 环境变量

本项目支持通过环境变量进行配置，建议在根目录创建 `.env` 文件：

- `SERVER_PORT` (默认 `8001`)：宿主机暴露的端口（用于 Docker）。
- `APP_PORT` (默认 `8000`)：服务内部监听端口。
- `WORKER_COUNT` (默认 `1`)：工作进程数。
- `USE_GPU` (默认 `True`)：是否启用显卡加速。
- `HOST_GPU_1`：宿主机显卡 ID（例如 `0`）。

示例 `.env` 内容：

```properties
# === 服务配置 ===
SERVER_PORT=8001
WORKER_COUNT=1

# === 显卡配置 ===
HOST_GPU_1=0

# === Paddle配置 ===
USE_GPU=True
```

> **提示**：在多 GPU 环境下，可以通过 `HOST_GPU_1` 指定使用的显卡。

---

## 🚀 API 接口说明

所有接口前缀默认为 `/api`（配置于 `main.py` 的 `root_path`）。

### 1. 健康检查 ❤️

- **方法**: `GET`
- **路径**: `/api/health`
- **返回示例**:

```json
{ "status": "ok", "gpu": true }
```

### 2. OCR 识别 🔍

- **方法**: `POST`
- **路径**: `/api/ocr`
- **参数**: `form-data`, key=`file`（支持 `.pdf`, `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`）
- **返回示例**:

```json
{
  "filename": "sample.pdf",
  "results": [
    {
      "page": 1, 
      "data": [{"text": "text content", "confidence": 0.98, "box": [[x,y], ...]}]
    }
  ]
}
```

示例 `curl` 请求:

```bash
curl -X 'POST' \
  'http://localhost:8001/api/ocr' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@xxx.png'
```

---

## 🛠️ 快速运行

### 方式一：本地直接运行

1. **安装依赖**

```powershell
python -m pip install -r requirements.txt
```

2. **启动服务**

```powershell
# 指定端口运行
$env:APP_PORT = 8000; python main.py
```

3. **访问文档**

- http://localhost:8000/docs

### 方式二：Docker 部署 (推荐)

```powershell
# 启动容器
docker-compose up --build -d
```

- http://localhost:@SERVER_PORT/docs

如果需要 GPU 支持，请在 docker-compose 或 docker run 中映射并启用 NVIDIA runtime，并设置 `USE_GPU=true` 环境变量。

---

## ⚠️ 注意事项

- **显卡兼容性**：PaddleOCR 和 CUDA 的配置依赖宿主环境，请确保 NVIDIA 驱动、CUDA、cuDNN 版本匹配。
- **共享内存**：在 Docker 中运行时，若 `WORKER_COUNT` > 1，请确保 `shm_size` 设置足够大（已在 `docker-compose.yml` 中默认设置为 32GB）。
- **生产建议**：建议在生产环境中添加 API 密钥校验、上传大小限制以及访问频率控制（Rate Limiting）。

---
