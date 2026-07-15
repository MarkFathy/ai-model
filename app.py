from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
import gc
from PIL import Image
import numpy as np

# Maximum allowed upload size (bytes) - protects against OOM from huge images
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MB

# Cap the longest image side before running OCR - big images are the
# single biggest driver of memory spikes/crashes on low-RAM instances
MAX_IMAGE_DIMENSION = 1600

ocr = None

def get_ocr():
    global ocr
    if ocr is None:
        from paddleocr import PaddleOCR
        # Initialize PaddleOCR for English (en) with MKLDNN disabled for stable deployment
        ocr = PaddleOCR(use_angle_cls=False, lang='en', enable_mkldnn=False)
    return ocr

app = FastAPI(title="PaddleOCR Hugging Face API")


@app.on_event("startup")
def load_model_on_startup():
    # Load the OCR model once, eagerly, at process startup instead of on
    # the first request. This avoids the first real request timing out
    # (and looking like a 502/503) while PaddleOCR downloads/loads its
    # weights, especially right after the free-tier instance wakes up.
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
        
        # Parse result (Supports both classic nested lists and new v6/Paddlex dictionary structures)
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
                    
                    # Convert numpy array coordinates to list
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
                        
                        # Convert numpy array coordinates to list
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
        # Explicitly release big objects and force garbage collection so
        # memory used by this request doesn't linger and accumulate
        # across requests on a low-RAM instance.
        try:
            del contents, image, img_np, result
        except NameError:
            pass
        gc.collect()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def home():
    return {"message": "PaddleOCR API is running. Send POST requests to /ocr"}

if __name__ == "__main__":
    import uvicorn
    # Hugging Face defaults to port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
