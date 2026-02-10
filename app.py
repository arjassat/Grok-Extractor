import streamlit as st
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageOps, ImageEnhance
import cv2
import numpy as np
import re
import tempfile
import os
import pandas as pd

st.set_page_config(page_title="Offline Invoice VAT Extractor v2", layout="wide")
st.title("🧾 Offline Invoice → VAT + Total Extractor (Improved)")
st.caption("100% local/offline: No API keys. Enhanced regex for your examples + better fallback.")

def preprocess_image(img):
    img = img.convert("L")  # grayscale
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Contrast(img).enhance(2.0)  # Increased contrast
    img = ImageEnhance.Sharpness(img).enhance(1.5)  # Added sharpness
    open_cv = np.array(img)
    open_cv = cv2.medianBlur(open_cv, 3)  # Noise reduction
    open_cv = cv2.threshold(open_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return Image.fromarray(open_cv)

def extract_text(file_bytes, filename):
    suffix = os.path.splitext(filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    text = ""
    try:
        if suffix == ".pdf":
            reader = PdfReader(tmp_path)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
            if len(text.strip()) < 50:  # Likely scanned PDF
                images = convert_from_path(tmp_path, dpi=500)  # Higher DPI for better OCR
                for img in images:
                    proc = preprocess_image(img)
                    text += pytesseract.image_to_string(proc, lang="eng", config='--psm 6') + "\n"  # PSM 6 for better layout
        else:  # image
            img = Image.open(tmp_path)
            proc = preprocess_image(img)
            text = pytesseract.image_to_string(proc, lang="eng", config='--psm 6')
    finally:
        os.unlink(tmp_path)
    return text.strip()

def parse_invoice(text):
    # Improved heuristic regex for South African invoices (based on your examples)
    total_patterns = [
        r'Total Payable:\s*R\s*([\d.,]+)',  # Amazon
        r'Total Amount\s*([\d.,]+)',        # Brights
        r'Grand Total:\s*R\s*([\d.,]+)',    # NATEC/General
        r'Total Incl\s*([\d.,]+)',          # NATEC
        r'Total\s+R\s*([\d.,]+)',           # Fallback, Le Creuset
        r'Total\s*([\d.,]+)',               # General
    ]
    
    vat_patterns = [
        r'VAT\s*\(15%\)\s*R\s*([\d.,]+)',   # Amazon
        r'Total VAT @ 15%\s*([\d.,]+)',     # Brights
        r'VAT\s*15%\s*R\s*([\d.,]+)',       # Variations
        r'VAT\s*R\s*([\d.,]+)',             # General
        r'Tax\s*([\d.,]+)',                 # NATEC
        r'Total VAT\s*([\d.,]+)',           # Fallback
    ]
    
    def find_match(patterns):
        for pat in patterns:
            matches = re.findall(pat, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                # Take the last/largest match as grand total/VAT
                try:
                    return float(matches[-1].replace(',', ''))
                except:
                    pass
        return 0.0
    
    total = find_match(total_patterns)
    vat = find_match(vat_patterns)
    
    # Improved fallback: Estimate VAT if no explicit but 15% or inclusive mentioned
    if vat == 0 and total > 0 and ('15%' in text.lower() or 'incl' in text.lower() or 'vat' in text.lower()):
        vat = round(total - (total / 1.15), 2)
    
    return vat, total

uploaded_files = st.file_uploader("Upload any number of PDFs or images", 
                                  type=["pdf","png","jpg","jpeg"], 
                                  accept_multiple_files=True)

if uploaded_files:
    results = []
    total_vat = 0.0
    total_amount = 0.0
    progress = st.progress(0)

    for i, up_file in enumerate(uploaded_files):
        progress.progress((i+1)/len(uploaded_files))
        file_bytes = up_file.read()
        text = extract_text(file_bytes, up_file.name)
        if not text:
            results.append({"File": up_file.name, "VAT (R)": 0.0, "Total (R)": 0.0, "Status": "OCR failed"})
            continue

        vat, amt = parse_invoice(text)
        status = "OK" if amt > 0 else "Parse failed - check text"

        results.append({"File": up_file.name, "VAT (R)": round(vat,2), "Total (R)": round(amt,2), "Status": status, "Extracted Text (snippet)": text[:200] + "..."})
        total_vat += vat
        total_amount += amt

    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("**Grand Total Amount**", f"R {total_amount:,.2f}")
    col2.metric("**Grand Total VAT**", f"R {total_vat:,.2f}")

    csv = df.to_csv(index=False).encode()
    st.download_button("Download CSV", csv, "invoices_summary.csv", "text/csv")

st.sidebar.info("""
**Improvements:**
- Better OCR preprocessing (sharper, less noise).
- More regex patterns for your specific invoices.
- Smarter VAT fallback for receipts like Le Creuset.
- Shows text snippet for debugging.
If still fails, run locally (streamlit run app.py) for faster testing.
""")
