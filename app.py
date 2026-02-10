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
from transformers import pipeline

# Disable decompression bomb check
Image.MAX_IMAGE_PIXELS = None

st.set_page_config(page_title="AI Invoice VAT Extractor", layout="wide")
st.title("🧾 AI-Powered Invoice → VAT + Total Extractor")
st.caption("Uses free local AI (LayoutLM) for extraction. No API keys. Works on PDFs/images.")

# Load AI model (caches after first load)
@st.cache_resource
def load_model():
    return pipeline("document-question-answering", model="impira/layoutlm-document-qa")

nlp = load_model()

def preprocess_image(img):
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    open_cv = np.array(img)
    open_cv = cv2.medianBlur(open_cv, 3)
    open_cv = cv2.threshold(open_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return Image.fromarray(open_cv)

def get_image_from_file(file_bytes, filename):
    suffix = os.path.splitext(filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    image = None
    try:
        if suffix == ".pdf":
            # Convert PDF to image (first page only to avoid large files)
            images = convert_from_path(tmp_path, dpi=300, first_page=1, last_page=1)
            image = preprocess_image(images[0])
        else:
            img = Image.open(tmp_path)
            image = preprocess_image(img)
    finally:
        os.unlink(tmp_path)
    return image

def parse_invoice_with_ai(image):
    if image is None:
        return 0.0, 0.0

    # Ask AI for total and VAT
    total_resp = nlp(image, "What is the grand total?")
    vat_resp = nlp(image, "What is the VAT amount?")

    total = 0.0
    vat = 0.0

    try:
        total = float(re.sub(r'[^\d.]', '', total_resp[0]['answer'])) if total_resp else 0.0
    except:
        pass

    try:
        vat = float(re.sub(r'[^\d.]', '', vat_resp[0]['answer'])) if vat_resp else 0.0
    except:
        pass

    # Fallback if VAT not found
    if vat == 0 and total > 0:
        vat = round(total * 0.15 / 1.15, 2)

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
        image = get_image_from_file(file_bytes, up_file.name)
        if image is None:
            results.append({"File": up_file.name, "VAT (R)": 0.0, "Total (R)": 0.0, "Status": "Processing failed"})
            continue

        vat, amt = parse_invoice_with_ai(image)
        status = "OK" if amt > 0 else "AI parse failed"

        results.append({"File": up_file.name, "VAT (R)": round(vat,2), "Total (R)": round(amt,2), "Status": status})
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
**Notes:**
- Uses LayoutLM AI model locally (free, no key).
- Handles scanned/digital PDFs/images.
- First run may take time to download model (~500MB).
- For multi-page PDFs, processes first page only.
- Lowered DPI to 300 to prevent errors.
- If VAT not detected, estimates at 15%.
""")
