# utils/document_extract.py
from __future__ import annotations

from typing import Optional
import io

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Best effort text extraction.
    Tries pdfplumber first, then pypdf.
    """
    text_parts: list[str] = []

    # Try pdfplumber
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t.strip():
                    text_parts.append(t)
        joined = "\n\n".join(text_parts).strip()
        if joined:
            return joined
    except Exception:
        pass

    # Fallback: pypdf
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for p in reader.pages:
            t = p.extract_text() or ""
            if t.strip():
                text_parts.append(t)
        return "\n\n".join(text_parts).strip()
    except Exception:
        return ""


def extract_text_from_docx(docx_bytes: bytes) -> str:
    """
    Extracts paragraph text plus OCR text from embedded images (tables/graphs often are images).
    """
    parts: list[str] = []

    # 1) Normal DOCX paragraph text
    try:
        from docx import Document  # type: ignore
        doc = Document(io.BytesIO(docx_bytes))
        for para in doc.paragraphs:
            if para.text and para.text.strip():
                parts.append(para.text.strip())
    except Exception:
        pass

    # 2) OCR embedded images (if utils/docx_ocr.py exists)
    try:
        from utils.docx_ocr import extract_images_from_docx, ocr_images_easyocr  # type: ignore
        img_bytes = extract_images_from_docx(docx_bytes)
        ocr_text = ocr_images_easyocr(img_bytes)
        if ocr_text and ocr_text.strip():
            parts.append("[OCR_EXTRACTED_FROM_IMAGES]")
            parts.append(ocr_text.strip())
    except Exception:
        pass

    return "\n".join(parts).strip()



def extract_text_from_upload(uploaded_file) -> str:
    """
    uploaded_file is Streamlit's UploadedFile.
    """
    name = (uploaded_file.name or "").lower()
    data = uploaded_file.getvalue()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(data)
    if name.endswith(".docx"):
        return extract_text_from_docx(data)

    return ""
