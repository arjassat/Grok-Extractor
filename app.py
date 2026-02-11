import streamlit as st
import pdfplumber
import pytesseract
from PIL import Image
import pandas as pd
import re
import numpy as np
import cv2

st.set_page_config(page_title="VAT Invoice Pro", layout="wide")

st.title("📊 VAT Invoice Extractor PRO (Free Cloud Edition)")
st.write("Mixed invoices (PDFs, scans, photos). Smart extraction with fallbacks.")

uploaded_files = st.file_uploader(
    "Upload invoices",
    type=["pdf", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ---------------- IMAGE OCR ----------------

def enhance_image(file):
    image = Image.open(file)
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    return gray

def extract_text_from_image(file):
    processed = enhance_image(file)
    return pytesseract.image_to_string(processed)

# ---------------- PDF EXTRACTION ----------------

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ---------------- MONEY HELPERS ----------------

def clean_amount(x):
    return float(x.replace(",", "").replace("R", "").strip())

def find_money_strings(text):
    return re.findall(r"\d[\d,]+\.\d{2}", text)

# ---------------- SMART EXTRACTION ----------------

def extract_financials(text):

    money_strings = find_money_strings(text)
    money_values = [clean_amount(m) for m in money_strings]

    if not money_values:
        return None, None, None, 0

    text_lower = text.lower()

    # --- VAT candidates ---
    vat_candidates = []
    for m_str, m_val in zip(money_strings, money_values):
        idx = text_lower.find(m_str)
        window = text_lower[max(0, idx-80):idx+80]

        if any(k in window for k in ["vat", "tax", "15%"]):
            vat_candidates.append(m_val)

    vat = None
    if vat_candidates:
        vat = min(vat_candidates)  # VAT usually smaller than total

    # --- TOTAL ---
    total = max(money_values)

    # --- EXCL ---
    excl = None
    if vat:
        excl = round(total - vat, 2)

    # --- FALLBACK VAT ---
    if not vat and total:
        vat = round(total * 0.15 / 1.15, 2)
        excl = round(total - vat, 2)

    # --- CONFIDENCE ---
    confidence = 40
    if vat and total:
        expected_vat = round(excl * 0.15, 2)
        if abs(expected_vat - vat) < 3:
            confidence += 40
        else:
            confidence += 20

    if vat in money_values:
        confidence += 10

    confidence = min(confidence, 100)

    return vat, total, excl, confidence

# ---------------- MAIN ----------------

if uploaded_files:

    results = []

    for file in uploaded_files:
        try:
            if file.type == "application/pdf":
                text = extract_text_from_pdf(file)
                if len(text.strip()) < 20:
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

        except Exception:
            results.append({
                "File Name": file.name,
                "Total Excl VAT": None,
                "VAT Amount": None,
                "Total Incl VAT": None,
                "Confidence %": 0
            })

    df = pd.DataFrame(results)

    st.subheader("📄 Extracted Results")
    st.dataframe(df)

    # FIX pandas warning + force numeric
    df["VAT Amount"] = pd.to_numeric(df["VAT Amount"], errors="coerce")
    df["Total Incl VAT"] = pd.to_numeric(df["Total Incl VAT"], errors="coerce")

    total_vat = df["VAT Amount"].sum()
    total_amount = df["Total Incl VAT"].sum()

    st.subheader("🧾 Totals")
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
