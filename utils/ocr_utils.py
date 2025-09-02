# utils/ocr_utils.py
import io
import numpy as np
from PIL import Image
import cv2

_EASYOCR_AVAILABLE = False
try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except Exception:
    _EASYOCR_AVAILABLE = False

try:
    import pytesseract
    from pytesseract import Output as pyt_Output
    _PYTESS_AVAILABLE = True
except Exception:
    _PYTESS_AVAILABLE = False

def get_ocr_reader(use_easyocr=True):
    """
    Returns an EasyOCR reader if available and user requested; otherwise None (we use pytesseract fallback).
    """
    if use_easyocr and _EASYOCR_AVAILABLE:
        # languages: english + hindi default; change if required
        return easyocr.Reader(['en','hi'], gpu=False)
    return None

def pil_to_cv(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv_to_pil(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def ocr_with_boxes(pil_img, reader=None):
    """
    Returns list of lines: {"text": str, "box": [x,y,w,h], "conf": float}
    """
    out = []
    if reader is not None:
        # EasyOCR expects numpy array
        arr = np.array(pil_img)
        res = reader.readtext(arr)
        for bbox, text, conf in res:
            xs = [int(p[0]) for p in bbox]
            ys = [int(p[1]) for p in bbox]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            out.append({"text": text, "box":[x0,y0,x1-x0,y1-y0], "conf": float(conf)})
        return out
    else:
        # pytesseract fallback
        if not _PYTESS_AVAILABLE:
            return []
        img = pil_img.convert("RGB")
        data = pytesseract.image_to_data(img, output_type=pyt_Output.DICT)
        n = len(data['text'])
        for i in range(n):
            txt = data['text'][i].strip()
            if not txt:
                continue
            conf = float(data['conf'][i]) if data['conf'][i] != '-1' else 0.0
            if conf < 30:
                continue
            x = int(data['left'][i]); y = int(data['top'][i]); w = int(data['width'][i]); h = int(data['height'][i])
            out.append({"text": txt, "box":[x,y,w,h], "conf": conf/100.0})
        return out
