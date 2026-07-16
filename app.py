import os
import gc
import io

# Disable PaddlePaddle 3.0.0 PIR executor and fall back to stable legacy executor on CPU
os.environ["FLAGS_enable_pir_api"] = "0"

# Restrict math libraries to 1 CPU thread to reduce RAM usage and prevent OOM crashes on free hosting
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# تحسين استراتيجية توزيع الذاكرة لـ Paddle لتجنب تراكمها في الـ RAM
os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"
os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = "0.0"

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import paddle  # استيراد مكتبة paddle للتحكم في تفريغ الذاكرة

# Maximum allowed upload size (bytes) - protects against OOM from huge images
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MB

# تقليل الحد الأقصى لأبعاد الصورة إلى 1000 لتوفير الذاكرة (مناسب جداً لكروت الشحن) [cite: 1]
MAX_IMAGE_DIMENSION = 1000 

ocr = None

def get_ocr():
    global ocr
    if ocr is None:
        from paddleocr import PaddleOCR
        # Initialize PaddleOCR for English (en) with CPU thread restrictions for low-RAM stability [cite: 1]
        ocr = PaddleOCR(
            use_angle_cls=False, 
            lang='en', 
            enable_mkldnn=False,
            cpu_threads=1,
            rec_batch_num=1
        )
    return ocr

app = FastAPI(title="PaddleOCR Hugging Face API")


@app.on_event("startup")
def load_model_on_startup():
    # Load the OCR model once, eagerly, at process startup [cite: 1]
    get_ocr()


def resize_if_needed(image: Image.Image) -> Image.Image:
    width, height = image.size
    longest_side = max(width, height)
    if longest_side > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / longest_side
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size, Image.LANCZOS)
    return image


@app.post("/ocr")
async def perform_ocr(file: UploadFile = File(...)):
    # تعريف المتغيرات في البداية لتجنب مشاكل الـ NameError في الـ finally
    contents = None
    image = None
    img_np = None
    result = None
    
    try:
        contents = await file.read()

        if len(contents) > MAX_UPLOAD_SIZE:
            return JSONResponse(
                content={"status": "error", "message": "File too large. Max size is 5MB."},
                status_code=413,
            )

        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = resize_if_needed(image)
        img_np = np.array(image)
        
        # Run PaddleOCR
        ocr_engine = get_ocr()
        result = ocr_engine.ocr(img_np)
        
        # Parse result (Supports both classic nested lists and new v6/Paddlex dictionary structures) [cite: 1]
        texts = []
        if result and len(result) > 0 and result[0] is not None:
            first_res = result[0]
            if isinstance(first_res, dict):
                rec_texts = first_res.get('rec_texts', [])
                rec_scores = first_res.get('rec_scores', [])
                rec_polys = first_res.get('rec_polys', [])
                for i in range(len(rec_texts)):
                    text = rec_texts[i]
                    confidence = rec_scores[i] if i < len(rec_scores) else 0.9
                    box = rec_polys[i] if i < len(rec_polys) else []
                    
                    # Convert numpy array coordinates to list [cite: 1]
                    if hasattr(box, 'tolist'):
                        box = box.tolist()
                    elif isinstance(box, np.ndarray):
                        box = box.tolist()
                        
                    texts.append({
                        "text": text,
                        "confidence": float(confidence),
                        "box": box
                    })
            elif isinstance(first_res, list):
                for line in first_res:
                    if isinstance(line, list) and len(line) >= 2:
                        box = line[0]
                        
                        # Convert numpy array coordinates to list [cite: 1]
                        if hasattr(box, 'tolist'):
                            box = box.tolist()
                        elif isinstance(box, np.ndarray):
                            box = box.tolist()
                            
                        text, confidence = line[1]
                        texts.append({
                            "text": text,
                            "confidence": float(confidence),
                            "box": box
                        })
        
        print(f"Recognized texts: {[t['text'] for t in texts]}")
        return JSONResponse(content={"status": "success", "data": texts})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
    
    finally:
        # مسح المتغيرات الضخمة فوراً من الذاكرة
        try:
            del contents
            del image
            del img_np
            del result
        except NameError:
            pass
        
        # 1. تفريغ الـ Cache الداخلي لمحرك PaddlePaddle لإرجاع الذاكرة لنظام التشغيل
        try:
            paddle.device.cuda.empty_cache()
        except Exception:
            pass
            
        # 2. إجبار الـ Garbage Collector الخاص بالبايثون على العمل فوراً
        gc.collect()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def home():
    return {"message": "PaddleOCR API is running. Send POST requests to /ocr"}

if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable (like on Railway) or default to 7860 [cite: 1]
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
