# utils/pdf_utils.py
import io
from PIL import Image
import fitz  # PyMuPDF


def load_pages_from_pdf(file_bytes, dpi=150):  # reduced DPI
    """
    Input: raw PDF bytes
    Output: list of PIL.Image pages (RGB)
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for i in range(len(doc)):
        pix = doc[i].get_pixmap(dpi=dpi, alpha=False)  # no alpha → faster
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(img)
    doc.close()
    return pages


def remove_pdf_metadata_bytes(pdf_bytes):
    """
    Remove PDF metadata and return cleaned bytes
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    doc.set_metadata({})
    out = doc.tobytes()
    doc.close()
    return out


def export_images_to_pdf(pil_images):
    """
    Return BytesIO containing a single PDF composed from pil_images
    """
    if not pil_images:
        return io.BytesIO()
    buf = io.BytesIO()
    pil_images[0].save(buf, format="PDF", resolution=150,
                       save_all=True, append_images=pil_images[1:])
    buf.seek(0)
    return buf
