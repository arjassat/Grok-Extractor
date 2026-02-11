import streamlit as st
import pdfplumber
import pytesseract
from PIL import Image
import re
import pandas as pd
import io
import numpy as np

st.set_page_config(page_title="VAT Invoice Extractor", layout="wide")

st.title("📄 VAT Invoice Extractor (South Africa)")
st.write("Upload PDF or Image invoices. The app will automatically extract VAT and Totals.")

uploaded_files = st.file_uploader(
    "Upload invoices (PDF, JPG, PNG)",
    type=["pdf", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# -------------------------------
# TEXT EXTRACTION
# -------------------------------

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_image(file):
    image = Image.open(file)
    image = image.convert("L")
    return pytesseract.image_to_string(image)

# -------------------------------
# SMART VAT + TOTAL DETECTION
# -------------------------------

def clean_amount(value):
    value = value.replace(",", "").replace("R", "").strip()
    return float(value)

def find_largest_amount(text):
    numbers = re.findall(r"\d[\d,]+\.\d{2}", text)
    if numbers:
        numbers = [clean_amount(n) for n in numbers]
        return max(numbers)
    return None

def extract_vat_and_totals(text):

    vat = None
    total = None
    excl = None

    # VAT patterns
    vat_patterns = [
        r"VAT.*?(\d[\d,]+\.\d{2})",
        r"Tax.*?(\d[\d,]+\.\d{2})",
        r"15%.*?(\d[\d,]+\.\d{2})"
    ]

    # Total patterns
    total_patterns = [
        r"Total incl.*?(\d[\d,]+\.\d{2})",
        r"Total amount.*?(\d[\d,]+\.\d{2})",
        r"Total.*?(\d[\d,]+\.\d{2})"
    ]

    # Excluding VAT
    excl_patterns = [
        r"Total excl.*?(\d[\d,]+\.\d{2})",
        r"Excl.*?(\d[\d,]+\.\d{2})"
    ]

    # Find VAT
    for pattern in vat_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            vat = clean_amount(match.group(1))
            break

    # Find Excl
    for pattern in excl_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            excl = clean_amount(match.group(1))
            break

    # Find Total
    for pattern in total_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            total = clean_amount(match.group(1))
            break

    # If no total found → assume largest number is total
    if not total:
        total = find_largest_amount(text)

    # If VAT missing but total & excl exist → calculate
    if not vat and total and excl:
        vat = round(total - excl, 2)

    # If VAT still missing but total exists → assume 15%
    if not vat and total:
        vat = round(total * 0.15 / 1.15, 2)

    return vat, total, excl


# -------------------------------
# MAIN PROCESSING
# -------------------------------

if uploaded_files:

    results = []

    for file in uploaded_files:

        if file.type == "application/pdf":
            text = extract_text_from_pdf(file)
        else:
            text = extract_text_from_image(file)

        vat, total, excl = extract_vat_and_totals(text)

        results.append({
            "File Name": file.name,
            "Total Excl VAT": excl,
            "VAT Amount": vat,
            "Total Incl VAT": total
        })

    df = pd.DataFrame(results)

    st.subheader("📊 Extracted Data Preview")
    st.dataframe(df)

    total_vat = df["VAT Amount"].fillna(0).sum()
    total_amount = df["Total Incl VAT"].fillna(0).sum()

    st.subheader("🧾 Totals Summary")
    st.success(f"Total VAT: R {total_vat:,.2f}")
    st.success(f"Total Invoice Amount: R {total_amount:,.2f}")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Results as CSV",
        csv,
        "vat_summary.csv",
        "text/csv"
    )

else:
    st.info("Upload invoices above to begin.")
