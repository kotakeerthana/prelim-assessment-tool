from __future__ import annotations

from typing import List
import io

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT


def _safe_text(s: str) -> str:
    return (s or "").strip()


def build_results_pdf_bytes(
    comments: str,
    conclusion: str,
    references: List[str],
    disclaimer: str,
    title: str = "Report Summary",
    page_size: str = "A4",
) -> bytes:
    """
    Returns PDF bytes containing Comments, Conclusion, References, Disclaimer.
    """
    buffer = io.BytesIO()

    pagesize = A4 if page_size.upper() == "A4" else letter
    doc = SimpleDocTemplate(
        buffer,
        pagesize=pagesize,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
        title=title,
    )

    styles = getSampleStyleSheet()

    # Make headings a bit stronger
    h_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        alignment=TA_LEFT,
        spaceAfter=10,
    )

    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        leading=14,
        spaceAfter=10,
    )

    small = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
        spaceBefore=8,
    )

    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Comments", h_style))
    story.append(Paragraph(_safe_text(comments) or "(None provided)", body))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Conclusion", h_style))
    story.append(Paragraph(_safe_text(conclusion) or "(None provided)", body))
    story.append(Spacer(1, 6))

    story.append(Paragraph("References", h_style))
    refs = references or []
    if refs:
        items = []
        for r in refs:
            txt = _safe_text(r)
            if not txt:
                continue
            items.append(ListItem(Paragraph(txt, body)))
        if items:
            story.append(ListFlowable(items, bulletType="1", leftIndent=18))
        else:
            story.append(Paragraph("(None provided)", body))
    else:
        story.append(Paragraph("(None provided)", body))

    story.append(Spacer(1, 10))
    story.append(Paragraph(disclaimer, small))

    doc.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
