# utils/redact_utils.py
import cv2
from PIL import Image, ImageDraw, ImageFilter
from .ocr_utils import cv_to_pil, pil_to_cv

def _rect_expand(box, img_shape, pad_frac=0.05):
    """
    Slightly expand a bounding box by pad_frac of width/height
    """
    x, y, w, h = box
    H, W = img_shape[:2]
    padx = int(w * pad_frac)
    pady = int(h * pad_frac)
    x0 = max(0, x - padx)
    y0 = max(0, y - pady)
    x1 = min(W, x + w + padx)
    y1 = min(H, y + h + pady)
    return [x0, y0, x1 - x0, y1 - y0]

def apply_redactions_cv(cv_img, detections, mode="blur"):
    """
    Redact boxes on a BGR OpenCV image.
    
    Parameters:
    - cv_img: BGR numpy array
    - detections: list of dicts with key 'box' -> [x,y,w,h]
    - mode: 'blur' or 'black'
    
    Returns:
    - PIL.Image (RGB) with redactions applied
    """
    out = cv_img.copy()
    H, W = out.shape[:2]

    for det in detections:
        box = det.get("box", [0, 0, 0, 0])
        if not box or box[2] == 0 or box[3] == 0:
            continue

        # expand slightly for nicer coverage
        x, y, w, h = box
        x0 = max(0, x - int(0.03 * w))
        y0 = max(0, y - int(0.03 * h))
        x1 = min(W, x + w + int(0.03 * w))
        y1 = min(H, y + h + int(0.03 * h))

        if mode == "black":
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
        else:
            roi = out[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            # dynamic blur kernel
            k = max(15, (max(1, (x1 - x0) // 10) // 2) * 2 + 1)
            blurred = cv2.GaussianBlur(roi, (k, k), 0)
            out[y0:y1, x0:x1] = blurred

    return cv_to_pil(out)


def apply_redactions_pil(pil_img, detections, mode="blur"):
    """
    Redact regions in a PIL image (fallback for Excel/DOCX screenshots)
    """
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)

    for det in detections:
        box = det.get("box", [0, 0, 0, 0])
        if not box or box[2] == 0 or box[3] == 0:
            continue
        x, y, w, h = box

        if mode == "black":
            draw.rectangle([x, y, x+w, y+h], fill="black")
        else:
            region = img.crop((x, y, x+w, y+h))
            region = region.filter(ImageFilter.GaussianBlur(radius=15))
            img.paste(region, (x, y))

    return img