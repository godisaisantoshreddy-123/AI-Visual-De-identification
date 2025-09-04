# utils/pdf_utils.py
import io
from functools import lru_cache
from PIL import Image
import fitz  # PyMuPDF

@lru_cache(maxsize=64)
def _pdf_page_count(pdf_bytes_hash: int):
    doc = fitz.open(stream=bytes.fromhex(hex(pdf_bytes_hash)[2:]), filetype="pdf")
    n = len(doc)
    doc.close()
    return n

def load_pages_from_pdf(file_bytes: bytes, dpi: int = 170, max_px: int = 1800):
    """
    Input: raw PDF bytes
    Output: list of PIL.Image pages (RGB), resized for speed
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for i in range(len(doc)):
        pix = doc[i].get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # downscale for speed if huge
        w, h = img.size
        scale = min(1.0, max_px / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        pages.append(img)
    doc.close()
    return pages

def remove_pdf_metadata_bytes(pdf_bytes: bytes):
    """Remove PDF metadata and return cleaned bytes (snake_case API)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    doc.set_metadata({})
    out = doc.tobytes()
    doc.close()
    return out

def export_images_to_pdf(pil_images):
    """Return BytesIO containing a single PDF composed from pil_images."""
    buf = io.BytesIO()
    if not pil_images:
        buf.seek(0)
        return buf
    pil_images[0].save(buf, format="PDF", save_all=True, append_images=pil_images[1:])
    buf.seek(0)
    return buf
