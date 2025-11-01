# app.py
import json
import os
import re
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any

import streamlit as st
import pandas as pd
from jinja2 import Template

from schemas import PatientInput, Vitals, Entities
from utils.entity_extraction import extract_entities, load_icd_map, parse_labs
from utils.triage_rules import collect_red_flags, naive_risk_bucket
from utils.pubmed import search_pubmed
from utils.db import log_record
from llm.llm_client import LLMClient
from prompts.prompts import BASE_SYSTEM_PROMPT, PROMPT_TEMPLATE
from utils.i18n import t, set_lang, get_lang, SUPPORTED, language_name


# =========================
# Page config
# =========================
st.set_page_config(page_title="Preliminary Assessment Tool", page_icon="🩺", layout="wide")


# =========================
# Helper functions
# =========================
def _localize_red_flags(raw_flags: list[str]) -> list[str]:
    # Map many raw phrasings to a single canonical code
    MAP = {
        # SBP hypotension
        "Hypotension (SBP < 90 mmHg)": "redflag.hypotension_sbp",
        "Hipotensión (PAS < 90 mmHg)": "redflag.hypotension_sbp",
        # DBP hypotension
        "Marked hypotension (DBP < 50 mmHg)": "redflag.hypotension_dbp",
        "Hipotensión marcada (PAD < 50 mmHg)": "redflag.hypotension_dbp",
        # Hypothermia
        "Hypothermia (<35.0 °C)": "redflag.hypothermia",
        "Hipotermia (<35.0 °C)": "redflag.hypothermia",
        # Sepsis pattern
        "Sepsis possible (vitals pattern)": "redflag.sepsis_pattern",
        # Tachycardia
        "Severe tachycardia (>130 bpm)": "redflag.tachycardia_marked",
        "Taquicardia grave (> 130 lpm)": "redflag.tachycardia_marked",
        "Taquicardia marcada (FC > 120 lpm)": "redflag.tachycardia_marked",
        # Tachypnea
        "Taquipnea marcada (FR > 30/min)": "redflag.tachypnea_marked",
        "Marked tachypnea (RR > 30/min)": "redflag.tachypnea_marked",
    }

    # Normalize → code → translate
    codes = []
    for f in raw_flags or []:
        code = MAP.get(f.strip())
        if code:
            codes.append(code)
        else:
            # fall back: keep original (will show as-is)
            codes.append(("__RAW__", f.strip()))

    # Deduplicate while preserving order
    seen = set()
    out = []
    for item in codes:
        key = item if isinstance(item, str) else item[1]
        if key in seen:
            continue
        seen.add(key)
        if isinstance(item, str):
            out.append(t(item, item))  # translate via i18n
        else:
            out.append(item[1])        # unknown string, show as-is
    return out

def _translate_llm_json_if_needed(json_obj: Dict[str, Any], target_lang_name: str, llm_client: LLMClient) -> Dict[str, Any]:
    # If target language is English, skip
    if str(target_lang_name).lower().startswith("english"):
        return json_obj

    # Try to detect English; even if unsure, we'll attempt translation once
    text_blob = " ".join(str(json_obj.get(k, "")) for k in [
        "overview", "key_findings", "differentials", "risk_assessment",
        "next_steps", "red_flags", "limitations", "references"
    ]).lower()
    english_cues = any(kw in text_blob for kw in [
        "presenting with", "estimated risk", "none elicited", "consider", "unlikely", "key findings"
    ])

    trans_prompt = f"""
    Return JSON only. Keep keys in English exactly:
    overview, key_findings, differentials, risk_assessment, next_steps, red_flags, limitations, references
    Translate ALL values into {target_lang_name}. Do not add or remove keys.

    JSON:
    {json.dumps(json_obj, ensure_ascii=False)}
    """.strip()

    out = llm_client.generate(trans_prompt)

    # Prefer direct JSON parse first
    try:
        return json.loads(out)
    except Exception:
        pass

    # Fallback: extract first JSON object
    m = re.search(r"\{.*\}", out, flags=re.S)
    if not m:
        return json_obj
    try:
        return json.loads(m.group(0))
    except Exception:
        return json_obj

def try_parse_llm_json(text: str) -> Optional[Dict[str, Any]]:
    """Find first JSON object in text and parse it; returns dict or None."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def compose_references_text(model_text: str, pubmed_refs: list, parsed_refs=None) -> str:
    """Merge model-provided refs + PubMed refs into a single text block."""
    out = []
    if parsed_refs:
        if isinstance(parsed_refs, list):
            out.extend([str(x).strip() for x in parsed_refs if str(x).strip()])
        elif isinstance(parsed_refs, str) and parsed_refs.strip():
            out.append(parsed_refs.strip())

    # Always append PubMed refs (titles — journal — year)
    if pubmed_refs:
        for r in pubmed_refs:
            title = (r.get("title") or "").strip()
            year = r.get("year") or ""
            journal = r.get("journal") or ""
            bits = [b for b in [title, journal, year] if b]
            if bits:
                out.append(" — ".join(bits))

    return "\n".join([x for x in out if x]).strip() or "(None provided)"


def default_sections(patient, ents, risk_bucket, red_flags):
    """Build safe, minimal sections if the model returns nothing."""
    # Overview
    overview = []
    # Localize the sex value
    SEX_LABEL = {
        "Female": t("sex.female", "Female"),
        "Male": t("sex.male", "Male"),
        "Intersex": t("sex.intersex", "Intersex"),
        "Other": t("sex.other", "Other"),
        "Unknown": t("sex.unknown", "Unknown"),
    }

    if patient.complaint:
        overview.append(t("default.presenting_with", "Presenting with") + " " + patient.complaint.lower())
    if patient.duration:
        dur = str(patient.duration).strip()
        overview.append("for " + (dur + " days" if dur.isdigit() else dur))
    if patient.severity_1_10:
        if not re.search(r"(severity|gravedad|severidad|తీవ్రత|गंभीरता)\s*\d+\s*/\s*10", (patient.complaint or ""), flags=re.I):
            overview.append("(" + t("form.severity") + f" {patient.severity_1_10}/10)")
    if not overview:
        overview.append(t("default.non_specific", "Presenting with a non-specific complaint."))
    demo = []
    if patient.age is not None:
        demo.append(t("default.age", "age") + f" {patient.age}")
    if patient.sex:
        demo.append(SEX_LABEL.get(patient.sex, patient.sex))
    if demo:
        overview.append(f"({', '.join(demo)})")
    overview_txt = " ".join(overview).strip()

    # Key findings
    kf = []
    if ents.symptoms:
        kf.append(t("default.symptoms", "Symptoms") + ": " + ", ".join(ents.symptoms))
    if patient.pmh:
        short_pmh = patient.pmh.strip()
        kf.append("PMH: " + (short_pmh[:200] + ("..." if len(short_pmh) > 200 else "")))
    vit = patient.vitals.model_dump() if patient.vitals else {}
    vit_list = []
    for k in ["heart_rate", "systolic_bp", "diastolic_bp", "temperature_c", "spo2", "respiratory_rate"]:
        v = vit.get(k)
        if v is not None:
            vit_list.append(f"{k}={v}")
    if vit_list:
        kf.append("Vitals: " + ", ".join(vit_list))
    key_findings = "\n".join(kf) or t("default.no_findings", "No key findings recorded.")

    # Differentials (generic starter text)
    differentials = t("default.differentials",
                  "Etiologies depend on location, character, and associated features. "
                  "Consider common, serious, and specialty-specific causes; refine with focused HPI, exam, and basic labs/imaging.")

    # Risk Assessment
    risk_assessment = t("misc.estimated_risk") + f": {bucket_label}."
    # Next steps (generic)
    steps = [
    t("default.step1", "Clarify symptom location, onset, triggers/relievers, and associated features."),
    t("default.step2", "Review vitals trend and repeat if abnormal."),
    t("default.step3", "Consider basic labs or imaging if red flags or persistent symptoms."),
    t("default.step4", "Escalate according to red flags and clinical judgment.")
    ]
    next_steps = "\n".join(f"- {s}" for s in steps)

    # Red flags
    red_flags_txt = ", ".join(red_flags) if red_flags else t("default.no_redflags", "None elicited.")

    # Limitations
    limitations = t("default.limitations",
                "Automated first-level summary. Not a diagnosis. Accuracy depends on input quality; "
                "important details (e.g., exam findings, ECG, labs) may change risk and differential.")

    return {
        "overview": overview_txt,
        "key_findings": key_findings,
        "differentials": differentials,
        "risk_assessment": risk_assessment,
        "next_steps": next_steps,
        "red_flags": red_flags_txt,
        "limitations": limitations,
    }


def build_pubmed_query(specialty: str, complaint: Optional[str], symptoms: List[str]) -> str:
    """Construct a tighter query that prefers guidelines/reviews and recent years."""
    terms = ", ".join(symptoms[:3]).strip()
    if not terms and complaint:
        terms = complaint.strip()
    base = f"({terms})" if terms else ""

    spec = (specialty or "").lower()
    if "cardio" in spec:
        base = f"{base} AND (chest pain OR acute coronary syndrome OR myocardial infarction)"
    elif "neuro" in spec:
        base = f"{base} AND (stroke OR ischemic stroke OR TIA)"
    elif "gastro" in spec:
        base = f"{base} AND (upper gastrointestinal bleeding OR hematemesis OR melena)"
    elif "endo" in spec:
        base = f"{base} AND (diabetic ketoacidosis OR hyperglycemic emergency)"
    elif "derma" in spec:
        base = f"{base} AND (cellulitis OR erysipelas)"
    elif "oncolog" in spec:
        base = f"{base} AND (febrile neutropenia OR neutropenic sepsis)"

    # Prefer guidelines/reviews and keep within the last ~10 years
    query = f"{base} AND (guideline[Publication Type] OR practice guideline[Publication Type] OR review[Publication Type]) AND 2015:3000[dp]"
    return query.strip(" AND ")


def curated_guidelines(specialty: str, complaint: Optional[str], symptoms: List[str]) -> List[str]:
    """Fallback guideline titles if PubMed returns nothing."""
    txt = " ".join([complaint or "", *symptoms]).lower()
    out: List[str] = []
    if "cardio" in specialty.lower():
        if "chest pain" in txt or "pain" in txt:
            out.append("AHA/ACC Guideline for the Evaluation and Diagnosis of Chest Pain (2021)")
            out.append("ESC Guidelines for non–ST-elevation acute coronary syndromes (recent edition)")
    if "neuro" in specialty.lower():
        if any(x in txt for x in ["weakness", "aphasia", "facial droop", "stroke"]):
            out.append("AHA/ASA Guideline for Early Management of Acute Ischemic Stroke (recent edition)")
    if "gastro" in specialty.lower():
        if any(x in txt for x in ["melena", "hematemesis", "black stools", "gi bleed"]):
            out.append("ACG Clinical Guideline: Upper Gastrointestinal and Ulcer Bleeding (recent edition)")
    if "endo" in specialty.lower():
        if any(x in txt for x in ["dka", "ketoacidosis", "ketones", "hyperglycemic"]):
            out.append("ADA Standards of Medical Care in Diabetes: Hyperglycemic Crises (current edition)")
    if "oncolog" in specialty.lower():
        if any(x in txt for x in ["fever", "neutropenia", "neutropenic"]):
            out.append("IDSA Guideline for Antimicrobial Agents in Neutropenic Patients (recent edition)")
    if "derma" in specialty.lower():
        if any(x in txt for x in ["cellulitis", "erysipelas", "skin infection"]):
            out.append("IDSA/Dermatology consensus on cellulitis management (recent edition)")
    return out


def vital_issues(v: Dict[str, Any]) -> List[str]:
    msgs = []
    if v.get("systolic_bp") is not None and v["systolic_bp"] < 70:
        msgs.append("SBP < 70 mmHg is extremely low — verify cuff/entry.")
    if v.get("diastolic_bp") is not None and v["diastolic_bp"] < 40:
        msgs.append("DBP < 40 mmHg is extremely low — verify cuff/entry.")
    if v.get("temperature_c") is not None and (v["temperature_c"] < 35.0 or v["temperature_c"] > 42.0):
        msgs.append("Temperature is outside physiologic range — verify °C units.")
    if v.get("spo2") is not None and v["spo2"] < 85:
        msgs.append("SpO₂ < 85% — recheck probe/reading immediately.")
    return msgs


# =========================
# Constants
# =========================
SPECIALTIES = [
    "cardiology", "oncology", "neurology", "endocrinology",
    "gastroenterology", "dermatology", "gp_im", "other"
]

def spec_label(code: str) -> str:
    return t(f"specialty.{code}", code)
# =========================
# Sidebar config
# =========================
st.sidebar.header("⚙️ Settings & Status")
provider = st.sidebar.selectbox("LLM Provider", ["gemini", "openai"], index=0)
use_llm_triage = st.sidebar.checkbox(
    "Use LLM triage (experimental)",
    value=False,
    help="Merge LLM-suggested red flags/risk with rule-based triage. Rules override if stricter."
)
st.sidebar.caption("Tip: Gemini has a generous free tier for prototyping.")
storage_mode = st.secrets.get("storage", {}).get("mode", "sqlite")
st.sidebar.write("Storage:", storage_mode)

# Language selector
with st.sidebar:
    st.caption("🌐")
    lang_options = list(SUPPORTED.keys())
    current_idx = lang_options.index(st.session_state.get("lang", "en")) if "lang" in st.session_state else 0
    chosen_lang = st.selectbox(
        t("nav.language", "Language"),
        options=lang_options,
        index=current_idx,
        format_func=lambda k: f"{SUPPORTED[k]} ({k})"
    )
    set_lang(chosen_lang)



# =========================
# Title
# =========================
st.markdown("# 🩺 " + t("app.title"))
st.caption("Educational demo. Not medical advice. Verify with clinical judgment and guidelines.")


# =========================
# Data / lookups
# =========================
icd_df = load_icd_map(os.path.join("data", "icd_map_sample.csv"))

# Toggle fallback (supports older Streamlit)
toggle = getattr(st, "toggle", st.checkbox)


# =========================
# Form
# =========================
with st.form("patient_form"):
    st.subheader(t("section.specialty"))
    specialty = st.selectbox(t("form.specialty_label"), SPECIALTIES, format_func=spec_label)

    st.subheader(t("section.demographics"))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        patient_id = st.text_input(t("form.patient_id"))
        age = st.number_input(t("form.age"), min_value=0, max_value=120, step=1, format="%d")
    with c2:
        sex = st.selectbox(t("form.sex"), ["Female", "Male", "Intersex", "Other", "Unknown"])
        race_ethnicity = st.text_input(t("form.race"))
    with c3:
        weight_kg = st.number_input(t("form.weight"), min_value=0.0, max_value=500.0, step=0.1)
        height_cm = st.number_input(t("form.height"), min_value=0.0, max_value=260.0, step=0.1)
    with c4:
        bmi = st.number_input(t("form.bmi"), min_value=0.0, max_value=100.0, step=0.1)

    st.subheader(t("section.hpi"))
    complaint = st.text_area(t("form.complaint"))
    duration = st.text_input(t("form.duration"))
    severity_1_10 = st.slider(t("form.severity"), min_value=1, max_value=10, value=5)
    associated_symptoms = st.text_area(t("form.assoc_symptoms"))

    st.subheader(t("section.pmh"))
    pmh = st.text_area(t("form.pmh"))
    surgical_history = st.text_area(t("form.surg"))
    medications = st.text_area(t("form.meds"))
    allergies = st.text_area(t("form.allergies"))
    family_history = st.text_area(t("form.family"))

    st.subheader(t("section.lifestyle"))
    smoking_status = st.text_input(t("form.smoking"))
    alcohol_consumption = st.text_input(t("form.alcohol"))
    illicit_drug_use = st.text_input(t("form.drugs"))

    st.subheader(t("section.vitals"))
    colv1, colv2, colv3, colv4, colv5, colv6 = st.columns(6)
    with colv1:
        hr = st.number_input(t("form.hr"), min_value=0, max_value=300, step=1)
    with colv2:
        sbp = st.number_input(t("form.sbp"), min_value=50, max_value=260, step=1)
    with colv3:
        dbp = st.number_input(t("form.dbp"), min_value=30, max_value=160, step=1)
    with colv4:
        rr = st.number_input(t("form.rr"), min_value=0, max_value=80, step=1)
    with colv5:
        tempc = st.number_input(t("form.temp"), min_value=30.0, max_value=45.0, step=0.1)
    with colv6:
        spo2 = st.number_input(t("form.spo2"), min_value=0, max_value=100, step=1)

    st.subheader(t("form.labs"))
    labs = st.text_area(t("form.labs"))
    imaging = st.text_area(t("form.imaging"))
    other_tests = st.text_area(t("form.other_tests"))

    st.subheader(t("misc.storage_privacy"))
    anonymize = toggle(
        "Anonymize & Save",
        value=False,
        help="If ON, will hash Patient ID and avoid storing direct identifiers.",
    )

    submitted = st.form_submit_button(t("section.submit"))


# =========================
# Submission handling
# =========================
if submitted:
    try:
        vitals = Vitals(
            heart_rate=hr or None,
            systolic_bp=sbp or None,
            diastolic_bp=dbp or None,
            respiratory_rate=rr or None,
            temperature_c=tempc or None,
            spo2=spo2 or None,
        )
        patient = PatientInput(
            patient_id=patient_id or None,
            specialty=specialty,  # type: ignore
            age=int(age) if age else None,
            sex=sex,
            race_ethnicity=race_ethnicity or None,
            weight_kg=float(weight_kg) if weight_kg else None,
            height_cm=float(height_cm) if height_cm else None,
            bmi=float(bmi) if bmi else None,
            complaint=complaint or None,
            duration=duration or None,
            severity_1_10=int(severity_1_10),
            associated_symptoms=associated_symptoms or None,
            pmh=pmh or None,
            surgical_history=surgical_history or None,
            medications=medications or None,
            allergies=allergies or None,
            family_history=family_history or None,
            smoking_status=smoking_status or None,
            alcohol_consumption=alcohol_consumption or None,
            illicit_drug_use=illicit_drug_use or None,
            vitals=vitals,
            labs=labs or None,
            imaging=imaging or None,
            other_tests=other_tests or None,
        )

        # --- Vital sign sanity checks & warnings ---
        warnings = []

        def warn_if(cond, msg):
            if cond:
                warnings.append(msg)

        # Typical adult ranges (adjust to your population if needed)
        warn_if(patient.vitals.heart_rate and patient.vitals.heart_rate > 100,
                "Taquicardia: FC > 100 lpm.")
        warn_if(patient.vitals.respiratory_rate and patient.vitals.respiratory_rate > 24,
                "Taquipnea: FR > 24/min.")
        warn_if(patient.vitals.systolic_bp and patient.vitals.systolic_bp < 90,
                "Hipotensión: PAS < 90 mmHg.")
        warn_if(patient.vitals.temperature_c and patient.vitals.temperature_c >= 38.0,
                "Fiebre: T ≥ 38.0 °C.")
        warn_if(patient.vitals.spo2 and patient.vitals.spo2 < 92,
                "Hipoxemia: SpO₂ < 92%.")

        if warnings:
            st.warning(" • " + "\n • ".join(warnings))

        # --- Entity extraction
        free_text = "\n".join(
            x for x in [
                patient.complaint, patient.associated_symptoms, patient.pmh,
                patient.medications, patient.allergies
            ] if x
        )
        ents_dict = extract_entities(free_text)
        ents = Entities(**ents_dict)

        st.subheader(t("misc.entities_extracted"))
        st.code(json.dumps(ents_dict, ensure_ascii=False, indent=2))

                # --- Parse labs (beta)
        labs_metrics = parse_labs(labs or "")

        # Normalize labs to a dict for triage rules
        if isinstance(labs_metrics, list):
            labs_struct = labs_metrics[0] if labs_metrics else {}
        elif isinstance(labs_metrics, dict):
            labs_struct = labs_metrics
        else:
            labs_struct = {}

        with st.expander(t("misc.parsed_labs")):
            st.json(labs_metrics)  # show whatever came back

        # --- Red flags & risk
        vitals_dict = patient.vitals.model_dump() if patient.vitals else {}
        red_flags = collect_red_flags(patient.specialty, vitals_dict, ents_dict.get("symptoms", []), labs_struct)
        risk_bucket = naive_risk_bucket(
            ents_dict,
            vitals_dict,
            complaint=patient.complaint,
            severity=patient.severity_1_10
        )


        bucket_label = {
            "Low": t("risk.low"),
            "Moderate": t("risk.moderate"),
            "High": t("risk.high"),
        }.get(risk_bucket, risk_bucket)

        # Vital sanity hints
        for msg in vital_issues(vitals_dict):
            st.warning(msg)

        rf_local = _localize_red_flags(red_flags)
        if rf_local:
            st.error("🚩 " + t("report.red_flags") + ": " + "; ".join(rf_local))

        bucket_label = {
            "Low": t("risk.low"),
            "Moderate": t("risk.moderate"),
            "High": t("risk.high"),
        }.get(risk_bucket, risk_bucket)
        st.info(t("misc.estimated_risk") + f": {bucket_label}")

        # --- PubMed evidence (smarter query + curated fallback)
        query = build_pubmed_query(patient.specialty, patient.complaint, ents.symptoms)
        refs = search_pubmed(query, max_results=3)
        st.subheader(" 📚 " + t("misc.evidence"))
        if refs:
            st.table(pd.DataFrame(refs))
        else:
            st.caption("No PubMed hits with current filters; using curated guideline references below.")
        curated_refs = curated_guidelines(patient.specialty, patient.complaint, ents.symptoms)

        # --- LLM prompt (strict JSON demanded in PROMPT_TEMPLATE)
        llm_client = LLMClient()
        llm_client.provider = provider

        condensed = {
            "specialty": patient.specialty,
            "complaint": patient.complaint,
            "duration": patient.duration,
            "severity": patient.severity_1_10,
            "sex": patient.sex,
            "age": patient.age,
            "vitals": vitals_dict,
            "entities": ents_dict,
            "red_flags": red_flags,
            "risk": risk_bucket,
        }
        prompt = PROMPT_TEMPLATE.render(
            system_prompt=BASE_SYSTEM_PROMPT,
            patient_json=json.dumps(condensed, ensure_ascii=False),
            specialty=patient.specialty,
            language_name=language_name()
     )


        with st.spinner("Generating summary..."):
            llm_text = llm_client.generate(prompt)

        with st.expander("LLM raw output"):
            st.code(llm_text or "(empty)")

        # --- Prefer JSON, fallback to default rule-based summary
        parsed = try_parse_llm_json(llm_text)
        if parsed:
            parsed = _translate_llm_json_if_needed(parsed, language_name(), llm_client)
            overview        = parsed.get("overview", "")
            key_findings    = parsed.get("key_findings", "")
            differentials   = parsed.get("differentials", "")
            risk_assessment = parsed.get("risk_assessment", "")
            next_steps      = parsed.get("next_steps", "")
            red_flags_txt   = parsed.get("red_flags", "")
            limitations     = parsed.get("limitations", "")
            references_txt  = compose_references_text("", refs, parsed.get("references"))
        else:
            defaults = default_sections(patient, ents, risk_bucket, red_flags)
            overview        = defaults["overview"]
            key_findings    = defaults["key_findings"]
            differentials   = defaults["differentials"]
            risk_assessment = defaults["risk_assessment"]
            next_steps      = defaults["next_steps"]
            red_flags_txt   = defaults["red_flags"]
            limitations     = defaults["limitations"]
            references_txt  = compose_references_text("", refs, None)

        # --- Merge LLM-suggested triage with rules (optional)
        def _norm_risk(s: Optional[str]):
            if not s:
                return None
            s = s.lower()
            if "high" in s:
                return "High"
            if "moderate" in s or "intermediate" in s:
                return "Moderate"
            if "low" in s:
                return "Low"
            return None

        if parsed and use_llm_triage:
            # Parse LLM "red_flags" (string or list)
            lf_raw = parsed.get("red_flags", "")
            if isinstance(lf_raw, str):
                parts = [p.strip() for p in re.split(r"[\n;,•]+", lf_raw) if p.strip()]
                llm_flags = parts
            elif isinstance(lf_raw, list):
                llm_flags = [str(x).strip() for x in lf_raw if str(x).strip()]
            else:
                llm_flags = []

            # Merge: rules + LLM
            red_flags = sorted(set(red_flags + llm_flags))

            # Risk: take the stricter one
            order = {"Low": 0, "Moderate": 1, "High": 2}
            llm_risk = _norm_risk(parsed.get("risk_assessment"))
            if llm_risk:
                risk_bucket = ["Low", "Moderate", "High"][max(order.get(risk_bucket, 0), order[llm_risk])]

            # Always reflect the final merged rule in the text shown
            rf_local = _localize_red_flags(red_flags)
            red_flags_txt = ", ".join(rf_local) if rf_local else t("default.no_redflags", "None elicited.")
            risk_assessment = t("misc.estimated_risk") + f": {bucket_label}."

        # References fallback to curated list if still empty
        if (not references_txt or references_txt.strip() == "(None provided)") and curated_refs:
            references_txt = "\n".join(curated_refs)

        specialty_label = spec_label(patient.specialty)

        # --- Render report via Jinja2 template
        tmpl_path = os.path.join("templates", "report_template.j2")
        with open(tmpl_path, "r", encoding="utf-8") as f:
            report_tmpl = Template(f.read())
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        specialty_label = spec_label(patient.specialty)
        report = report_tmpl.render(
            title=t("report.title"),
            specialty_label=specialty_label,
            generated_at=now,
            head_overview=t("report.overview"),
            head_key_findings=t("report.key_findings"),
            head_differentials=t("report.differentials"),
            head_risk_assessment=t("report.risk_assessment"),
            head_next_steps=t("report.next_steps"),
            head_red_flags=t("report.red_flags"),
            head_limitations=t("report.limitations"),
            head_references=t("report.references"),
            head_specialty=t("report.specialty"),
            head_generated_at=t("report.generated_at"),
            overview=overview,
            key_findings=key_findings,
            differentials=differentials,
            risk_assessment=risk_assessment,
            next_steps=next_steps,
            red_flags=red_flags_txt,
            limitations=limitations,
            references=references_txt,
        )
        st.subheader("📝 Generated Report")
        st.markdown(report, unsafe_allow_html=True)

        # --- Optional logging
        if anonymize:
            pid_stored = hashlib.sha256((patient.patient_id or "").encode()).hexdigest()[:10] if patient.patient_id else None
        else:
            pid_stored = patient.patient_id

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "patient_id": pid_stored,
            "specialty": patient.specialty,
            "age": patient.age,
            "sex": patient.sex,
            "race_ethnicity": patient.race_ethnicity,
            "weight": patient.weight_kg,
            "height": patient.height_cm,
            "bmi": patient.bmi,
            "complaint": patient.complaint,
            "duration": patient.duration,
            "severity": patient.severity_1_10,
            "assoc_symptoms": patient.associated_symptoms,
            "pmh": patient.pmh,
            "surg_hx": patient.surgical_history,
            "meds": patient.medications,
            "allergies": patient.allergies,
            "fam_hx": patient.family_history,
            "smoking": patient.smoking_status,
            "alcohol": patient.alcohol_consumption,
            "drugs": patient.illicit_drug_use,
            "vitals": json.dumps(vitals_dict),
            "labs": patient.labs,
            "imaging": patient.imaging,
            "other_tests": patient.other_tests,
            "entities": json.dumps(ents_dict),
            "pubmed_refs": json.dumps(refs),
            "report_raw": llm_text,
            "report_rendered": report,
        }
        try:
            log_record(record)
            st.success("Saved (opt-in logging).")
        except Exception as e:
            st.warning(f"Could not save: {e}")

    except Exception as e:
        st.error(f"Error: {e}")
