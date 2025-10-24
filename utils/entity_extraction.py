# utils/entity_extraction.py
from typing import Dict, List
import os
import re
import pandas as pd

# ---- Optional lightweight NER via scispaCy (auto-fallback to regex) ----
try:
    import spacy
    _NLP = spacy.load("en_core_sci_sm")  # scispaCy small model
    _SCISPACY = True
except Exception:
    _NLP = None
    _SCISPACY = False

# ---- Minimal ICD-like map loader (sample only) ----
def load_icd_map(path: str) -> pd.DataFrame:
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame({"term": [], "icd": []})

def map_to_icd(terms: List[str], icd_df: pd.DataFrame) -> List[Dict[str, str]]:
    if icd_df.empty:
        return [{"term": t, "icd": ""} for t in sorted(set(terms))]
    out: List[Dict[str, str]] = []
    for t in sorted(set(terms)):
        row = icd_df[icd_df["term"].str.lower() == t.lower()]
        icd = row["icd"].iloc[0] if not row.empty else ""
        out.append({"term": t, "icd": icd})
    return out

# ---- Regex heuristics (fast, no GPU) ----
# Expand symptom coverage beyond the basic hints
_SYMPTOM_PATTERNS: List[tuple[re.Pattern, str]] = [
    (re.compile(r"\babdominal (pain|ache)\b", re.I), "abdominal pain"),
    (re.compile(r"\bchest (pain|pressure|tightness)\b", re.I), "chest pain"),
    (re.compile(r"\b(short(ness)? of breath|dyspnea|sob)\b", re.I), "shortness of breath"),
    (re.compile(r"\bnausea\b", re.I), "nausea"),
    (re.compile(r"\b(vomiting|emesis)\b", re.I), "vomiting"),
    (re.compile(r"\bdiarrh(oe)a\b", re.I), "diarrhea"),
    (re.compile(r"\bfever|febrile\b", re.I), "fever"),
    (re.compile(r"\bheadache\b", re.I), "headache"),
    (re.compile(r"\bsyncope|faint(ing)?\b", re.I), "syncope"),
    (re.compile(r"\bdizz(y|iness)\b", re.I), "dizziness"),
    (re.compile(r"\bpalpitations?\b", re.I), "palpitations"),
    (re.compile(r"\bpolyuria\b", re.I), "polyuria"),
    (re.compile(r"\bpolydipsia\b", re.I), "polydipsia"),
    (re.compile(r"\b(fruity|ketotic)\s+breath\b", re.I), "fruity breath"),
    (re.compile(r"\bkussmaul\b", re.I), "kussmaul respirations"),
    (re.compile(r"\bmelena|black stools?\b", re.I), "melena"),
    (re.compile(r"\bhematemesis\b", re.I), "hematemesis"),
    (re.compile(r"\bhives|urticaria\b", re.I), "hives"),
    (re.compile(r"\bwheez(e|ing)\b", re.I), "wheezing"),
    (re.compile(r"\bthroat (tightness|closure)\b", re.I), "throat tightness"),
    (re.compile(r"\bcalf pain\b", re.I), "calf pain"),
    (re.compile(r"\bfatigue|tired(ness)?\b", re.I), "fatigue"),
    (re.compile(r"\bpain\b", re.I), "pain"),
]

# Known meds to catch (demo) + dose patterns
_KNOWN_MEDS = {"insulin", "metformin", "atorvastatin", "aspirin", "apixaban", "albuterol", "omeprazole", "amoxicillin"}
_MED_DOSE_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_-]{2,})\s+(?:\d+(?:\.\d+)?)\s*(mg|mcg|g|units?)\b", re.I)

_ALLERGY_FREE_RE = re.compile(r"\bNKDA\b", re.I)
_ALLERGY_HINT_RE = re.compile(r"\ballergic to ([^.;,\n]+)", re.I)

def _dedupe(xs: List[str]) -> List[str]:
    return sorted(set(x.strip() for x in xs if x and x.strip()))

def extract_entities(text: str) -> Dict[str, List[str]]:
    """Light, no-GPU entity extraction for demo."""
    text = (text or "").strip()
    if not text:
        return {"symptoms": [], "conditions": [], "medications": [], "allergies": []}

    symptoms: List[str] = []
    conditions: List[str] = []
    medications: List[str] = []
    allergies: List[str] = []

    # 1) scispaCy (if available): collect surface entities, then bucket via regex hints
    if _SCISPACY and _NLP is not None:
        try:
            doc = _NLP(text)
            for ent in doc.ents:
                et = ent.text.strip()
                # try to map to symptoms first
                matched = False
                for pat, canon in _SYMPTOM_PATTERNS:
                    if pat.search(et):
                        symptoms.append(canon); matched = True; break
                if not matched:
                    conditions.append(et)
        except Exception:
            pass

    # 2) Regex patterns for symptoms (robust fallback & supplement)
    for pat, canon in _SYMPTOM_PATTERNS:
        if pat.search(text):
            symptoms.append(canon)

    # 3) Medications
    for m in _KNOWN_MEDS:
        if re.search(rf"\b{re.escape(m)}\b", text, re.I):
            medications.append(m)
    for m in _MED_DOSE_RE.findall(text):
        medications.append(m[0].lower())

    # 4) Allergies
    if not _ALLERGY_FREE_RE.search(text):
        for m in _ALLERGY_HINT_RE.findall(text):
            allergies.append(m.strip().lower())

    # 5) Very light "conditions" signals
    if re.search(r"\b(type\s*[12]\s*diabet|diabetes)\b", text, re.I):
        conditions.append("diabetes")
    if re.search(r"\b(h(igh )?blood pressure|hypertension|htn)\b", text, re.I):
        conditions.append("hypertension")
    if re.search(r"\b(hyperlipid|dyslipid|hld)\b", text, re.I):
        conditions.append("hyperlipidemia")

    return {
        "symptoms": _dedupe(symptoms),
        "conditions": _dedupe(conditions),
        "medications": _dedupe(medications),
        "allergies": _dedupe(allergies),
    }

# ---- Minimal lab parser used by triage across specialties ----
_MMOL_L_TO_MG_DL = {"glucose": 18.0}

def parse_labs(labs_text: str) -> Dict[str, float | bool | None]:
    """
    Extracts a few key labs commonly referenced in triage:
    - glucose_mg_dl (auto converts from mmol/L if detected)
    - hco3 (bicarbonate)
    - anion_gap
    - anion_gap_flag_high (bool)
    - ketones_positive (bool)
    """
    t = labs_text or ""
    out: Dict[str, float | bool | None] = {
        "glucose_mg_dl": None,
        "hco3": None,
        "anion_gap": None,
        "anion_gap_flag_high": False,
        "ketones_positive": False,
    }

    # Glucose
    m = re.search(r"\b(glucose|blood\s*sugar|bg)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(mg/?d?l|mmol/?l)?", t, re.I)
    if m:
        val = float(m.group(2))
        unit = (m.group(3) or "").lower()
        if "mmol" in unit:
            val *= _MMOL_L_TO_MG_DL["glucose"]
        out["glucose_mg_dl"] = val

    # HCO3
    m = re.search(r"\b(hco3|bicarb(onate)?)\s*[:=]?\s*(\d+(?:\.\d+)?)\b", t, re.I)
    if m:
        out["hco3"] = float(m.group(3))

    # Anion gap
    m = re.search(r"\b(anion\s*gap|ag)\s*[:=]?\s*(\d+(?:\.\d+)?)\b", t, re.I)
    if m:
        out["anion_gap"] = float(m.group(2))
    if re.search(r"\b(anion\s*gap|ag)\s+(high|elevated)\b", t, re.I):
        out["anion_gap_flag_high"] = True

    # Ketones
    if re.search(r"\bketones?\s*(positive|\+)\b", t, re.I) or re.search(r"\b(ketonemia|ketonuria)\b", t, re.I):
        out["ketones_positive"] = True

    return out
