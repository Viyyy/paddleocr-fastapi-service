import os
import cv2
import numpy as np
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from paddleocr import PaddleOCR
from typing import List

# 从环境变量读取配置
# 显卡控制主要通过 Docker 传递 CUDA_VISIBLE_DEVICES 环境变量
USE_GPU = os.getenv("USE_GPU", "True").lower() == "true"
GPU_MEM = int(os.getenv("GPU_MEM", "4000")) # 显存限制预留

app = FastAPI(title="PaddleOCR Service", version="2.7.3", root_path="/api")

# 初始化 PaddleOCR 引擎
# use_angle_cls: 是否加载分类模型
# lang: 语言，默认中文 ch
# use_gpu: 是否使用 GPU
print(f"Initializing PaddleOCR... GPU={USE_GPU}")
ocr_engine = PaddleOCR(
    use_angle_cls=True, 
    lang="ch", 
    use_gpu=USE_GPU, 
    # gpu_mem=GPU_MEM,
    show_log=False
)

def read_image_file(file_bytes: bytes) -> np.ndarray:
    """将字节流转换为 OpenCV 图像格式"""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def parse_pdf(file_bytes: bytes) -> List[np.ndarray]:
    """将 PDF 字节流转换为图像列表"""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    images = []
    for page in doc:
        # zoom=2 为 2倍分辨率，提高 OCR 识别率
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        # 将 pixmap 转为 numpy 数组 (Height, Width, Channel)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        # 如果是 RGB 转 BGR (OpenCV 默认是 BGR)
        if pix.n >= 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
    return images

@app.post("/ocr")
async def ocr_predict(file: UploadFile = File(...)):
    filename = file.filename.lower()
    content = await file.read()
    
    images_to_process = []

    try:
        if filename.endswith(".pdf"):
            images_to_process = parse_pdf(content)
        elif filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            images_to_process = [read_image_file(content)]
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Only PDF and Images allowed.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

    results = []
    
    for idx, img in enumerate(images_to_process):
        # paddleocr 输入是 numpy array
        # result 结构: [[[[x1,y1],[x2,y2]...], (text, confidence)], ...]
        res = ocr_engine.ocr(img, cls=True)
        
        page_result = []
        if res and res[0]:
            for line in res[0]:
                box = line[0]
                text = line[1][0]
                score = line[1][1]
                page_result.append({
                    "text": text,
                    "confidence": float(score),
                    "box": box
                })
        
        results.append({
            "page": idx + 1,
            "data": page_result
        })

    return {"filename": file.filename, "results": results}

@app.get("/health")
def health_check():
    return {"status": "ok", "gpu": USE_GPU}

if __name__ == "__main__":
    import uvicorn
    # 获取端口和 Host 配置
    port = int(os.getenv("APP_PORT", 8000))
    host = "0.0.0.0"
    print(f"Starting server on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port)