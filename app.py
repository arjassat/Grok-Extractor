import streamlit as st
import pdfplumber
import pytesseract
from PIL import Image
import pandas as pd
import re
import io
import numpy as np
import cv2

st.set_page_config(page_title="VAT Invoice Pro", layout="wide")

st.title("📊 VAT Invoice Extractor PRO (Free Cloud Edition)")
st.write("Upload mixed invoices (PDF, scans, photos). Smart extraction + validation.")

uploaded_files = st.file_uploader(
    "Upload invoices",
    type=["pdf", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ---------------------------
# IMAGE ENHANCEMENT
# ---------------------------

def enhance_image(file):
    image = Image.open(file)
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    return thresh

def extract_text_from_image(file):
    processed = enhance_image(file)
    return pytesseract.image_to_string(processed)

# ---------------------------
# PDF EXTRACTION
# ---------------------------

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        total_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                # Focus more on last 40% of document
                if i >= total_pages * 0.6:
                    text += page_text + "\n"
    return text

# ---------------------------
# MONEY DETECTION
# ---------------------------

def clean_amount(x):
    return float(x.replace(",", "").replace("R", "").strip())

def extract_money_values(text):
    matches = re.findall(r"\d[\d,]+\.\d{2}", text)
    return [clean_amount(m) for m in matches]

# ---------------------------
# CONTEXT SCORING ENGINE
# ---------------------------

def score_value(text, value_str):
    score = 0
    index = text.find(value_str)
    if index == -1:
        return score

    window = text[max(0, index-100): index+100].lower()

    if "vat" in window: score += 50
    if "tax" in window: score += 40
    if "total" in window: score += 30
    if "incl" in window: score += 20
    if "excl" in window: score += 20
    if "amount due" in window: score += 25

    return score

# ---------------------------
# SMART EXTRACTION
# ---------------------------

def extract_financials(text):

    money_strings = re.findall(r"\d[\d,]+\.\d{2}", text)
    money_values = [clean_amount(m) for m in money_strings]

    if not money_values:
        return None, None, None, 0

    scored = []

    for m_str, m_val in zip(money_strings, money_values):
        s = score_value(text, m_str)
        scored.append((m_val, s))

    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)

    vat = None
    total = None
    excl = None

    # Highest scored value assumed VAT candidate
    if scored_sorted:
        vat_candidate = scored_sorted[0][0]
        vat = vat_candidate

    # Assume largest value is total
    total = max(money_values)

    # Calculate excl if possible
    if vat and total:
        excl = round(total - vat, 2)

    # Mathematical validation
    confidence = 50

    if vat and total:
        calc_vat = round(total * 0.15 / 1.15, 2)
        if abs(calc_vat - vat) < 2:
            confidence += 40

    if excl and vat:
        if abs((excl * 0.15) - vat) < 2:
            confidence += 20

    return vat, total, excl, min(confidence, 100)

# ---------------------------
# MAIN PROCESSING
# ---------------------------

if uploaded_files:

    results = []

    for file in uploaded_files:

        try:
            if file.type == "application/pdf":
                text = extract_text_from_pdf(file)
                if not text.strip():
                    text = extract_text_from_image(file)
            else:
                text = extract_text_from_image(file)

            vat, total, excl, confidence = extract_financials(text)

            results.append({
                "File Name": file.name,
                "Total Excl VAT": excl,
                "VAT Amount": vat,
                "Total Incl VAT": total,
                "Confidence %": confidence
            })

        except Exception as e:
            results.append({
                "File Name": file.name,
                "Total Excl VAT": None,
                "VAT Amount": None,
                "Total Incl VAT": None,
                "Confidence %": 0
            })

    df = pd.DataFrame(results)

    st.subheader("📄 Extracted Data")
    st.dataframe(df)

    total_vat = df["VAT Amount"].fillna(0).sum()
    total_amount = df["Total Incl VAT"].fillna(0).sum()

    st.subheader("🧾 Summary Totals")
    st.success(f"Total VAT: R {total_vat:,.2f}")
    st.success(f"Total Invoice Total: R {total_amount:,.2f}")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download CSV",
        csv,
        "vat_invoice_summary.csv",
        "text/csv"
    )

else:
    st.info("Upload invoices to begin.")
