import re
import cv2
import numpy as np
from pyzbar.pyzbar import decode

# ---------- Optional spaCy NER (names, addresses, orgs) ----------
_NLP = None
def get_nlp(enable_ner: bool):
    """Lazily load spaCy model if enabled; return None if disabled or load fails."""
    global _NLP
    if not enable_ner:
        return None
    if _NLP is not None:
        return _NLP
    try:
        import spacy
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            _NLP = spacy.load("en_core_web_sm")
        return _NLP
    except Exception:
        return None


# ---------- Regex patterns ----------
PATTERNS = {
    "EMAIL": re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "PHONE": re.compile(r"(?<!\d)(?:\+?\d{1,3}[-\s]?)?(?:\(?\d{3}\)?[-\s]?){1}\d{3}[-\s]?\d{4}(?!\d)"),
    "AADHAAR": re.compile(r"(?<!\d)(?:\d{4}[-\s]?\d{4}[-\s]?\d{4})(?!\d)"),
    "PAN": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    "GSTIN": re.compile(r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b"),
    "IFSC": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "NPI": re.compile(r"\b\d{10}\b"),
    "PASSPORT": re.compile(r"\b[A-PR-WYa-pr-wy][0-9]{7,8}\b"),
    "DL_GENERIC": re.compile(r"\b[A-Z0-9]{6,15}\b"),
    "CREDIT_CARD": re.compile(r"(?:(?:\d{4}[- ]?){3}\d{4})"),
    "CVV": re.compile(r"(?<!\d)\d{3}(?!\d)"),
    "BANK_ACC": re.compile(r"(?<!\d)\d{9,18}(?!\d)"),
    "DOB": re.compile(r"\b(?:DOB|Date of Birth|Birth Date)\s*[:\-]?\s*(?:\d{1,2}[/\-]\d{1,2}[/\-](?:19|20)\d{2}|[A-Za-z]{3,9}\s+\d{1,2},\s+(?:19|20)\d{2})\b", re.I),
    "ICD10": re.compile(r"\b[A-TV-Z][0-9][0-9AB](?:\.[0-9A-TV-Z]{1,4})?\b"),
    "IP": re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b"),
    "MAC": re.compile(r"\b(?:[0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}\b"),
    "URL": re.compile(r"https?://[^\s]+"),
    "INSURANCE_ID": re.compile(r"\b(?:Member\s*ID|Policy\s*No\.?|Subscriber\s*ID|Payer\s*ID|Group\s*No\.?)\s*[:\-]?\s*[A-Z0-9\-]{5,20}\b", re.I),
}

SIGNATURE_HINTS = ["signature", "sign", "signee"]
ADDRESS_HINT_WORDS = [
    "address", "addr.", "street", "st.", "road", "rd.", "lane", "ln.",
    "avenue", "ave.", "boulevard", "blvd", "city", "state", "zip", "pincode", "pin"
]


# ---------- Regex findings ----------
def regex_entities(text: str):
    findings = []
    for label, pat in PATTERNS.items():
        for m in pat.finditer(text):
            findings.append({"label": label, "span": [m.start(), m.end()], "matched": m.group(0)})
    for line in text.splitlines():
        low = line.lower()
        if any(k in low for k in ADDRESS_HINT_WORDS) or re.search(r"\d{1,5}\s+[A-Za-z0-9.\- ]+,\s*[A-Za-z.\- ]+[, ]+\w{2}\s+\d{4,6}", line):
            if len(line.strip()) > 6:
                findings.append({"label": "ADDRESS", "span": [0, 0], "matched": line.strip()[:200]})
    return findings


def align_text_findings_to_boxes(text_findings, ocr_lines):
    boxes = []
    for tf in text_findings:
        target = tf["matched"]
        placed = False
        for line in ocr_lines:
            if target and target in line["text"]:
                boxes.append({"label": tf["label"], "box": line["box"], "matched": target})
                placed = True
                break
        if not placed:
            t_clean = re.sub(r"\W+", "", target.lower())
            for line in ocr_lines:
                l_clean = re.sub(r"\W+", "", line["text"].lower())
                if t_clean and t_clean in l_clean and len(t_clean) >= 4:
                    boxes.append({"label": tf["label"], "box": line["box"], "matched": target})
                    placed = True
                    break
        if not placed:
            boxes.append({"label": tf["label"], "box": [0, 0, 0, 0], "matched": target})
    return boxes


# ---------- Face detection ----------
_face_cascade = None
def get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _face_cascade

def detect_faces(cv_img):
    if cv_img is None or cv_img.size == 0:
        return []
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    faces = get_face_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    return [{"type": "FACE", "box": [int(x), int(y), int(w), int(h)]} for (x, y, w, h) in faces]


# ---------- Signature regions ----------
def find_signature_regions(cv_img, ocr_lines):
    if cv_img is None or cv_img.size == 0:
        return []
    h, w = cv_img.shape[:2]
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    dil = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    regions = []
    for line in ocr_lines:
        text = line.get("text", "").lower()
        if any(k in text for k in SIGNATURE_HINTS):
            x, y, w0, h0 = line["box"]
            y1 = min(h, y + h0 + int(3 * h0))
            x0 = max(0, x - int(0.2 * w0)); x1 = min(w, x + w0 + int(0.2 * w0))
            roi = dil[y:y1, x0:x1]
            cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if 500 < area < 60000:
                    rx, ry, rw, rh = cv2.boundingRect(c)
                    aspect = rw / max(1, rh)
                    if aspect > 2.5:
                        regions.append({"type": "SIGNATURE", "box": [int(x0 + rx), int(y + ry), int(rw), int(rh)]})
    return regions


# ---------- QR code detection ----------
def detect_qr_regions(cv_img):
    results = []
    if cv_img is None or cv_img.size == 0:
        return results
    detections = decode(cv_img)
    for d in detections:
        (x, y, w, h) = d.rect
        results.append({
            "type": "QRCODE",
            "box": [x, y, w, h],
            "matched": d.data.decode("utf-8", errors="ignore")
        })
    return results


# ---------- NER ----------
def ner_findings(text: str, ocr_lines, enable_ner: bool):
    nlp = get_nlp(enable_ner)
    if not nlp or not text:
        return []
    doc = nlp(text)
    ents = []
    for e in doc.ents:
        if e.label_ in ("PERSON", "GPE", "ORG"):
            ents.append({"label": e.label_, "matched": e.text})
    return align_text_findings_to_boxes(ents, ocr_lines)


# ---------- Unified pipeline ----------
def detect_sensitive_regions(cv_img, ocr_lines, full_text, enable_ner: bool = False):
    results = []

    # Regex-based entities
    regex_findings = regex_entities(full_text)
    results.extend(align_text_findings_to_boxes(regex_findings, ocr_lines))

    # Named entities (spaCy)
    if enable_ner:
        results.extend(ner_findings(full_text, ocr_lines, enable_ner))

    # Faces
    results.extend(detect_faces(cv_img))

    # Signatures
    results.extend(find_signature_regions(cv_img, ocr_lines))

    # QR Codes
    results.extend(detect_qr_regions(cv_img))

    return results
