# utils/report_generation.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import re

from llm.llm_client import LLMClient

def _extract_sections_from_report_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to extract COMENTARIO / CONCLUSIÓN / Referencia sections directly from the report text.
    Works well for reports like the sample PDF.
    """
    if not text or not text.strip():
        return None

    t = text.replace("\r", "\n")

    # Accept English and Spanish labels
    comment_labels = r"(COMENTARIO|COMMENT|COMMENTS)"
    conclusion_labels = r"(CONCLUSI[ÓO]N|CONCLUSION)"
    reference_labels = r"(REFERENCIA|REFERENCIAS|REFERENCE|REFERENCES)"

    # Grab comment block
    m_comment = re.search(
        rf"{comment_labels}\s*:\s*(.+?)(?=\n\s*{conclusion_labels}\s*:|\n\s*{reference_labels}\s*:|$)",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Grab conclusion block
    m_conc = re.search(
        rf"{conclusion_labels}\s*:\s*(.+?)(?=\n\s*{reference_labels}\s*:|$)",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Grab references block
    m_refs = re.search(
        rf"{reference_labels}\s*:\s*(.+)$",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not (m_comment or m_conc or m_refs):
        return None

    comments = (m_comment.group(2) if m_comment else "").strip()
    conclusion = (m_conc.group(2) if m_conc else "").strip()

    refs: List[str] = []
    if m_refs:
        refs_text = m_refs.group(2).strip()
        # split references by lines starting with numbers or bullets
        for line in re.split(r"\n+", refs_text):
            s = line.strip()
            if not s:
                continue
            # keep numbered references, or any non-empty line
            refs.append(s)

        # clean obvious trailing notes
        refs = [r for r in refs if not r.lower().startswith("nota:")]
    
    cleaned: list[str] = []
    for r in refs:
        s = r.strip()

        # Remove lines that are only repeated numbers like "1 1 1 1" or "2 2"
        if re.fullmatch(r"(\d+\s*){2,}", s):
            continue

        # Remove single digit lines that are just numbering
        if re.fullmatch(r"\d+", s):
            continue

        cleaned.append(s)

    refs = cleaned

    return {
        "comments": comments,
        "conclusion": conclusion,
        "references": refs,
        "source": "extracted_from_report",
    }


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """
    Tries to find the first JSON object in the model output.
    """
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    # Direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Find a JSON object inside text
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def detect_language_code(text: str, fallback: str = "en") -> str:
    """
    Uses langdetect to guess language. Maps into your app language codes.
    """
    try:
        from langdetect import detect  # type: ignore
        raw = (detect(text or "") or "").lower()
    except Exception:
        return fallback

    if raw.startswith("es"):
        return "es"
    if raw.startswith("fr"):
        return "fr"
    if raw.startswith("pt"):
        return "pt"
    if raw.startswith("zh"):
        return "zh"
    if raw.startswith("ar"):
        return "ar"
    if raw.startswith("hi"):
        return "hi"

    return "en"


def generate_comment_conclusion_references(
    report_text_any_lang: str,
    llm_client: LLMClient,
    target_lang_name: str,
) -> Dict[str, Any]:
    target_lang_name = target_lang_name or "English"
    """
    Returns a dict with keys: comments, conclusion, references (list of strings).
    Output is in target_lang_name.
    """
    if not report_text_any_lang or not report_text_any_lang.strip():
        return {}
    extracted = _extract_sections_from_report_text(report_text_any_lang)
    if extracted:
        extracted_comments = (extracted.get("comments") or "").strip()
        extracted_conclusion = (extracted.get("conclusion") or "").strip()
        extracted_refs = extracted.get("references", [])

    # Only skip the LLM if we actually extracted narrative sections
        if extracted_comments or extracted_conclusion:
            return {
                "comments": extracted_comments,
                "conclusion": extracted_conclusion,
                "feedback": "",  # add feedback key for UI consistency
                "references": extracted_refs,
                "raw": "",
            }
    # If only references exist, keep them but continue to LLM
        refs_from_report = extracted_refs
    else:
        refs_from_report = []

    # Prevent extremely long reports from causing model failures
    max_chars = 12000
    report_for_llm = report_text_any_lang[:max_chars]

    prompt = f"""
You are a clinician assistant. You are given a diagnostic report text.
Produce:

1) Comments: 
   - Interpret any available findings (symptom trends, protocol, criteria, tables if present).
   - If numeric results or measurements are missing, explicitly say so.
   - Do NOT invent values.
2) Conclusion:    
   - Provide a cautious, limited conclusion based only on what is present.
   - If key data (e.g., H2/CH4 values, timing of peaks) is missing, state that the conclusion is limited.
3) Feedback / Next steps:
   - Clearly list what additional information would be needed to make a stronger interpretation
     (e.g., missing gas measurements, prep compliance, medications, antibiotics, timing).
4) References: a list of 2 to 4 reputable guideline, consensus, or review style references.
   If the report contains explicit references, extract them and keep them.
   If references exist in the report, copy them exactly without translating.

Important rules:
- Do not claim a definitive diagnosis if the report language says it is not definitive.
- If details are missing, say so.
- Do not invent journal details you are not sure about.
- Prefer guideline names over fake citations.
- Keep output concise and structured.

Return JSON only with exactly this shape:
{{
  "comments": "string",
  "conclusion": "string",
  "feedback": "string",
  "references": ["string", "string"]
}}

Write the JSON values in {target_lang_name}.

Report text:
{report_for_llm}
""".strip()

    raw = llm_client.generate(prompt)
    if not raw or not raw.strip():
        return {
            "comments": "",
            "conclusion": "",
            "feedback": "",
            "references": refs_from_report,
            "raw": "(empty LLM response: check API key, provider, or model errors)",
        }

    
    parsed = _extract_json_block(raw)

    # If JSON parsing failed, retry with a stricter prompt
    if not isinstance(parsed, dict):
        retry_prompt = f"""
Return ONLY valid JSON. No prose. No markdown.

Required format:
{{
  "comments": "string",
  "conclusion": "string",
  "feedback": "string",
  "references": ["string", "string"]
}}

Write the JSON values in {target_lang_name}.
Use ONLY information present in the report. If key numeric results are missing, say that explicitly.
Do not invent a definitive diagnosis.

Report:
{report_for_llm}
""".strip()

        raw_retry = llm_client.generate(retry_prompt)
        parsed = _extract_json_block(raw_retry)
        raw = raw_retry

        if not isinstance(parsed, dict):
            return {
                "comments": "",
                "conclusion": "",
                "feedback": "",
                "references": refs_from_report,
                "raw": raw_retry,
            }



    comments = str(parsed.get("comments", "")).strip()
    conclusion = str(parsed.get("conclusion", "")).strip()
    feedback = str(parsed.get("feedback", "")).strip()
    refs: List[str] = []
    r = parsed.get("references", [])
    if isinstance(r, list):
        for item in r:
            s = str(item).strip()
            if s:
                refs.append(s)
    elif isinstance(r, str) and r.strip():
        refs.append(r.strip())

    return {
        "comments": comments,
        "conclusion": conclusion,
        "feedback": feedback,
        "references": refs if refs else refs_from_report,
        "raw": raw,
    }