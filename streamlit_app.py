# app.py
import os
import io
import json
import zipfile
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# Handle Spacy model for cloud deployment
# -----------------------------
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    import spacy
    nlp = spacy.load("en_core_web_sm")

# -----------------------------
# Local imports
# -----------------------------
from utils.ocr_utils import get_ocr_reader, pil_to_cv, cv_to_pil, ocr_with_boxes
from utils.detect_utils import (
    get_all_text_findings,
    align_text_findings_to_boxes,
    detect_faces,
    find_signature_regions,
    mask_excel_cells_and_render
)
from utils.redact_utils import apply_redactions_cv, apply_redactions_pil
from utils.pdf_utils import load_pages_from_pdf, remove_pdf_metadata_bytes, export_images_to_pdf
from utils.doc_utils import extract_text_from_docx_stream

# -----------------------------
# Setup output folders
# -----------------------------
OUTPUT_DIR = "outputs"
RED_PDFS = os.path.join(OUTPUT_DIR, "redacted_pdfs")
RED_ZIPS = os.path.join(OUTPUT_DIR, "redacted_zips")
RED_LOGS = os.path.join(OUTPUT_DIR, "logs")
UPLOAD_DIR = "uploads"

for d in (RED_PDFS, RED_ZIPS, RED_LOGS, UPLOAD_DIR):
    os.makedirs(d, exist_ok=True)

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="AI Visual De-ID", layout="wide")
st.title("🔒 AI Visual De-identification – Privacy by Design")

with st.expander("ℹ️ Instructions", expanded=True):
    st.markdown("""
    Upload images (**jpg/png/jpeg**), **PDFs, DOCX, XLSX**.  
    The app will:
    - OCR the document
    - Detect **PII/PHI** via regex and heuristics
    - Detect **faces** and **signatures**
    - Let you toggle detections per-box
    - Redact with **blur** or **black box**
    - Export results as **PDF, ZIP (images + JSON log)**
    """)

# -----------------------------
# Options
# -----------------------------
redact_method = st.radio("Redaction method", options=["blur", "black"], index=0, horizontal=True)
enable_faces = st.checkbox("Detect Faces", value=True)
enable_signatures = st.checkbox("Detect Signatures", value=True)
use_easyocr = st.checkbox("Use EasyOCR (recommended)", value=True)
st.write("")

uploads = st.file_uploader(
    "Upload files (images, pdf, docx, xlsx). You can select multiple.",
    accept_multiple_files=True,
    type=["png","jpg","jpeg","pdf","docx","xlsx"]
)

process = st.button("🚀 Process & Redact")

# -----------------------------
# Helpers
# -----------------------------
def timestamp():
    return datetime.now().strftime("%Y%m%dT%H%M%S")

# Threaded image/PDF page processor
def process_image_threaded(pil_img, reader, enable_faces=True, enable_signatures=True):
    cv_img = pil_to_cv(pil_img)
    lines = ocr_with_boxes(pil_img, reader)
    full_text = "\n".join([l["text"] for l in lines])
    text_findings = get_all_text_findings(full_text)
    text_boxes = align_text_findings_to_boxes(text_findings, lines)
    face_boxes = detect_faces(cv_img) if enable_faces else []
    sig_boxes = find_signature_regions(cv_img, lines) if enable_signatures else []
    merged = text_boxes + face_boxes + sig_boxes
    redacted_pil = apply_redactions_cv(cv_img, merged, mode=redact_method)
    return redacted_pil, merged

def process_images_in_batch(images, reader, enable_faces=True, enable_signatures=True):
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_image_threaded, img, reader, enable_faces, enable_signatures) for img in images]
        for f in futures:
            red_pil, merged = f.result()
            results.append((red_pil, merged))
    return results

# -----------------------------
# Main logic
# -----------------------------
if process and uploads:
    reader = get_ocr_reader(use_easyocr)
    all_logs = []
    redacted_images_global = []

    progress_bar = st.progress(0)
    total_files = len(uploads)

    with st.spinner("Processing documents..."):
        for file_idx, file in enumerate(uploads, start=1):
            st.markdown(f"### 📄 Processing `{file.name}`")
            name = file.name
            ext = os.path.splitext(name)[1].lower()

            # IMAGE HANDLER
            if ext in [".png", ".jpg", ".jpeg"]:
                pil = Image.open(file).convert("RGB")
                red_pil, merged = process_image_threaded(pil, reader, enable_faces, enable_signatures)
                st.image(red_pil, caption="Redacted preview", use_column_width=True)
                redacted_images_global.append(red_pil)
                all_logs.append({"file": name, "type":"image", "detections": merged, "redacted": merged, "mode": redact_method, "timestamp": timestamp()})
                progress_bar.progress(file_idx / total_files)

            # PDF HANDLER
            elif ext == ".pdf":
                raw = file.read()
                stripped = remove_pdf_metadata_bytes(raw)
                pages = load_pages_from_pdf(stripped)
                batch_results = process_images_in_batch(pages, reader, enable_faces, enable_signatures)
                for pno, (red_pil, merged) in enumerate(batch_results, start=1):
                    st.image(red_pil, caption=f"Page {pno} redacted preview", use_column_width=True)
                    redacted_images_global.append(red_pil)
                    all_logs.append({"file": name, "type":"pdf","page":pno,"detections":merged,"redacted":merged,"mode":redact_method,"timestamp":timestamp()})
                    progress_bar.progress((file_idx-1 + pno/len(pages))/total_files)

            # DOCX HANDLER
            elif ext == ".docx":
                text = extract_text_from_docx_stream(file)
                text_findings = get_all_text_findings(text)
                img = Image.new("RGB", (1654, 2339), "white")
                draw = ImageDraw.Draw(img)
                try: font = ImageFont.truetype("DejaVuSans.ttf", 18)
                except: font = ImageFont.load_default()
                y = 40
                for line in text.splitlines():
                    draw.text((40,y), line[:2000], fill="black", font=font)
                    y += 22
                    if y > 2280: break
                merged = [{"label": m["label"], "matched": m["matched"], "box":[40, y-40, 200, 24]} for m in text_findings]
                red_pil = apply_redactions_pil(img, merged, mode=redact_method)
                st.image(red_pil, caption="DOCX redacted preview", use_column_width=True)
                redacted_images_global.append(red_pil)
                all_logs.append({"file": name, "type":"docx", "detections": merged,"redacted": merged,"mode": redact_method,"timestamp":timestamp()})
                progress_bar.progress(file_idx / total_files)

            # XLSX HANDLER
            elif ext == ".xlsx":
                sheets_rendered = mask_excel_cells_and_render(file)
                for idx, (sheetname, pil_img, detections) in enumerate(sheets_rendered):
                    red_pil = apply_redactions_pil(pil_img, detections, mode=redact_method)
                    st.image(red_pil, caption=f"Sheet {sheetname} redacted preview", use_column_width=True)
                    redacted_images_global.append(red_pil)
                    all_logs.append({"file": name, "type":"xlsx","sheet": sheetname,"detections": detections,"redacted": detections,"mode":redact_method,"timestamp":timestamp()})
                progress_bar.progress(file_idx / total_files)

            else:
                st.warning(f"Unsupported file type: {ext}")

    # Export outputs
    if redacted_images_global:
        ts = timestamp()
        pdf_bytes = export_images_to_pdf(redacted_images_global)
        pdf_name = f"redacted_{ts}.pdf"
        pdf_path = os.path.join(RED_PDFS, pdf_name)
        with open(pdf_path, "wb") as f: f.write(pdf_bytes.getvalue())
        st.success(f"Redacted PDF created: {pdf_name}")
        st.download_button("⬇️ Download Redacted PDF", data=open(pdf_path,"rb"), file_name=pdf_name, mime="application/pdf")

        zip_name = f"redacted_bundle_{ts}.zip"
        zip_path = os.path.join(RED_ZIPS, zip_name)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for i, img in enumerate(redacted_images_global):
                b = io.BytesIO()
                img.save(b, format="PNG")
                z.writestr(f"redacted_{i+1:03d}.png", b.getvalue())
            log_name = f"redaction_log_{ts}.json"
            z.writestr(log_name, json.dumps(all_logs, indent=2))
        st.download_button("⬇️ Download ZIP (images + log)", data=open(zip_path,"rb"), file_name=zip_name, mime="application/zip")

        log_path = os.path.join(RED_LOGS, f"redaction_log_{ts}.json")
        with open(log_path, "w") as f: json.dump(all_logs, f, indent=2)
        st.success("✅ All done — check outputs/ folder or download above.")
else:
    st.info("Upload files and click 'Process & Redact' to start.")
