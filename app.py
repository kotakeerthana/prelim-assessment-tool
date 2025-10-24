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


# =========================
# Page config
# =========================
st.set_page_config(page_title="Preliminary Assessment Tool", page_icon="🩺", layout="wide")


# =========================
# Helper functions
# =========================
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
    if patient.complaint:
        overview.append(f"Presenting with {patient.complaint.lower()}")
    if patient.duration:
        dur = str(patient.duration).strip()
        overview.append("for " + (dur + " days" if dur.isdigit() else dur))
    if patient.severity_1_10:
        overview.append(f"(severity {patient.severity_1_10}/10)")
    if not overview:
        overview.append("Presenting with a non-specific complaint.")
    demo = []
    if patient.age is not None:
        demo.append(f"age {patient.age}")
    if patient.sex:
        demo.append(patient.sex)
    if demo:
        overview.append(f"({', '.join(demo)})")
    overview_txt = " ".join(overview).strip()

    # Key findings
    kf = []
    if ents.symptoms:
        kf.append("Symptoms: " + ", ".join(ents.symptoms))
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
    key_findings = "\n".join(kf) or "No key findings recorded."

    # Differentials (generic starter text)
    differentials = (
        "Etiologies depend on location, character, and associated features. "
        "Consider common, serious, and specialty-specific causes; refine with focused HPI, exam, and basic labs/imaging."
    )

    # Risk Assessment
    risk_assessment = f"Estimated risk bucket: {risk_bucket}."

    # Next steps (generic)
    steps = [
        "Clarify symptom location, onset, triggers/relievers, and associated features.",
        "Review vitals trend and repeat if abnormal.",
        "Consider basic labs or imaging if red flags or persistent symptoms.",
        "Escalate according to red flags and clinical judgment."
    ]
    next_steps = "\n".join(f"- {s}" for s in steps)

    # Red flags
    red_flags_txt = ", ".join(red_flags) if red_flags else "None elicited."

    # Limitations
    limitations = (
        "Automated first-level summary. Not a diagnosis. Accuracy depends on input quality; "
        "important details (e.g., exam findings, ECG, labs) may change risk and differential."
    )

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
    "Cardiology", "Oncology", "Neurology", "Endocrinology",
    "Gastroenterology", "Dermatology", "General Practice/Internal Medicine", "Other"
]


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


# =========================
# Title
# =========================
st.markdown("# 🩺 Preliminary Assessment Tool (MVP+)")
st.caption("Educational demo — not medical advice. Verify with clinical judgment & guidelines.")


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
    st.subheader("I. Specialty Selection")
    specialty = st.selectbox("Specialty", SPECIALTIES)

    st.subheader("II. Patient Demographics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        patient_id = st.text_input("Patient ID (optional)")
        age = st.number_input("Age", min_value=0, max_value=120, step=1, format="%d")
    with c2:
        sex = st.selectbox("Sex", ["Female", "Male", "Intersex", "Other", "Unknown"])
        race_ethnicity = st.text_input("Race/Ethnicity")
    with c3:
        weight_kg = st.number_input("Weight (kg)", min_value=0.0, max_value=500.0, step=0.1)
        height_cm = st.number_input("Height (cm)", min_value=0.0, max_value=260.0, step=0.1)
    with c4:
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, step=0.1)

    st.subheader("III. Complaints")
    complaint = st.text_area("Nature of Complaint")
    duration = st.text_input("Duration (e.g., 3 days, 2 weeks)")
    severity_1_10 = st.slider("Severity (1–10)", min_value=1, max_value=10, value=5)
    associated_symptoms = st.text_area("Associated Symptoms (free text)")

    st.subheader("IV. Medical History")
    pmh = st.text_area("Past Medical History")
    surgical_history = st.text_area("Surgical History")
    medications = st.text_area("Medications")
    allergies = st.text_area("Allergies")
    family_history = st.text_area("Family History")

    st.subheader("V. Social History")
    smoking_status = st.text_input("Smoking Status")
    alcohol_consumption = st.text_input("Alcohol Consumption")
    illicit_drug_use = st.text_input("Illicit Drug Use")

    st.subheader("VI. Physical Examination Findings")
    colv1, colv2, colv3, colv4, colv5, colv6 = st.columns(6)
    with colv1:
        hr = st.number_input("HR (bpm)", min_value=0, max_value=300, step=1)
    with colv2:
        sbp = st.number_input("SBP (mmHg)", min_value=50, max_value=260, step=1)
    with colv3:
        dbp = st.number_input("DBP (mmHg)", min_value=30, max_value=160, step=1)
    with colv4:
        rr = st.number_input("RR (/min)", min_value=0, max_value=80, step=1)
    with colv5:
        tempc = st.number_input("Temp (°C)", min_value=30.0, max_value=45.0, step=0.1)
    with colv6:
        spo2 = st.number_input("SpO₂ (%)", min_value=0, max_value=100, step=1)

    st.subheader("VII. Initial Diagnostic Test Results")
    labs = st.text_area("Laboratory Results")
    imaging = st.text_area("Imaging Findings")
    other_tests = st.text_area("Other relevant tests")

    st.subheader("Storage & Privacy")
    anonymize = toggle(
        "Anonymize & Save",
        value=False,
        help="If ON, will hash Patient ID and avoid storing direct identifiers.",
    )

    submitted = st.form_submit_button("Generate Summary")


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

        # --- Entity extraction
        free_text = "\n".join(
            x for x in [
                patient.complaint, patient.associated_symptoms, patient.pmh,
                patient.medications, patient.allergies
            ] if x
        )
        ents_dict = extract_entities(free_text)
        ents = Entities(**ents_dict)

        st.success("Entities extracted")
        st.json(ents_dict)

        # --- Parse labs (beta)
        labs_metrics = parse_labs(labs or "")
        with st.expander("Parsed labs (beta)"):
            st.json(labs_metrics)

        # --- Red flags & risk
        vitals_dict = patient.vitals.model_dump() if patient.vitals else {}
        red_flags = collect_red_flags(patient.specialty, vitals_dict, ents_dict.get("symptoms", []), labs_metrics)
        risk_bucket = naive_risk_bucket(vitals_dict, red_flags)

        # Vital sanity hints
        for msg in vital_issues(vitals_dict):
            st.warning(msg)

        if red_flags:
            st.error("🚩 Red Flags: " + "; ".join(red_flags))
        st.info(f"Estimated Risk: {risk_bucket}")

        # --- PubMed evidence (smarter query + curated fallback)
        query = build_pubmed_query(patient.specialty, patient.complaint, ents.symptoms)
        refs = search_pubmed(query, max_results=3)
        st.subheader(" 📚 Evidence (PubMed)")
        if refs:
            st.table(pd.DataFrame(refs))
        else:
            st.caption("No PubMed hits with current filters; using curated guideline references below.")
        curated_refs = curated_guidelines(patient.specialty, patient.complaint, ents.symptoms)

        # --- LLM prompt (strict JSON demanded in PROMPT_TEMPLATE)
        llm_client = LLMClient()
        llm_client.provider = provider  # override from sidebar

        condensed = patient.model_dump()
        condensed["vitals"] = vitals_dict
        condensed["entities"] = ents_dict
        condensed["red_flags"] = red_flags
        condensed["risk"] = risk_bucket
        condensed_json = json.dumps(condensed, indent=2)

        prompt = PROMPT_TEMPLATE.render(
            system_prompt=BASE_SYSTEM_PROMPT,
            patient_json=condensed_json,
            specialty=patient.specialty,
        )

        with st.spinner("Generating summary..."):
            llm_text = llm_client.generate(prompt)

        with st.expander("LLM raw output"):
            st.code(llm_text or "(empty)")

        # --- Prefer JSON, fallback to default rule-based summary
        parsed = try_parse_llm_json(llm_text)
        if parsed:
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
            red_flags_txt = ", ".join(red_flags) if red_flags else "None elicited."
            risk_assessment = f"Estimated risk bucket: {risk_bucket}."

        # References fallback to curated list if still empty
        if (not references_txt or references_txt.strip() == "(None provided)") and curated_refs:
            references_txt = "\n".join(curated_refs)

        # --- Render report via Jinja2 template
        tmpl_path = os.path.join("templates", "report_template.j2")
        with open(tmpl_path, "r", encoding="utf-8") as f:
            report_tmpl = Template(f.read())
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        report = report_tmpl.render(
            title="Preliminary Assessment Report (Non-diagnostic)",
            specialty=patient.specialty,
            generated_at=now,
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
        st.markdown(report)

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
