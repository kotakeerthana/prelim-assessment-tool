# utils/triage_rules.py
from typing import Dict, Any, List, Callable, Tuple

Rule = Tuple[str, Callable[[Dict[str, Any], List[str], Dict[str, Any] | None], bool]]

def _v(v: Dict[str, Any], k: str, default=None):
    return v.get(k, default)

# -------- absolute (apply to everyone) --------
ABSOLUTE_RULES: List[Rule] = [
    ("Hypotension (SBP < 90 mmHg)",        lambda vit, s, labs: (_v(vit,"systolic_bp",999) < 90)),
    ("Marked hypotension (DBP < 50 mmHg)", lambda vit, s, labs: (_v(vit,"diastolic_bp",999) < 50)),
    ("Hypoxia (SpO₂ < 90%)",               lambda vit, s, labs: (_v(vit,"spo2",100) < 90)),
    ("Severe tachycardia (>130 bpm)",      lambda vit, s, labs: (_v(vit,"heart_rate",0) > 130)),
    ("Severe bradycardia (<40 bpm)",       lambda vit, s, labs: (_v(vit,"heart_rate",999) < 40)),
    ("Hyperthermia (>39.5 °C)",            lambda vit, s, labs: (_v(vit,"temperature_c",0) > 39.5)),
    ("Hypothermia (<35.0 °C)",             lambda vit, s, labs: (_v(vit,"temperature_c",999) < 35.0)),
]

# -------- lab-driven helpers (used across specialties) --------
def _suspect_dka(vit: Dict[str,Any], s: List[str], labs: Dict[str,Any] | None) -> bool:
    if not labs: return False
    g = labs.get("glucose_mg_dl"); hco3 = labs.get("hco3")
    ag = labs.get("anion_gap"); ag_high = bool(labs.get("anion_gap_flag_high"))
    ket = bool(labs.get("ketones_positive"))
    rr = _v(vit,"respiratory_rate",0); sset = set(s)
    poly = ("polyuria" in sset) or ("polydipsia" in sset) or ("fruity breath" in sset) or ("kussmaul respirations" in sset)
    return ((g and g >= 250) or poly) and ((hco3 and hco3 < 18) or ket or ag_high or (ag and ag >= 12) or rr >= 24)

def _suspect_sepsis(vit: Dict[str,Any], s: List[str], labs: Dict[str,Any] | None) -> bool:
    # demo: tachy >100 + fever >=38.3 or hypothermia + hypotension
    hr = _v(vit,"heart_rate",0); t = _v(vit,"temperature_c",36.8); sbp=_v(vit,"systolic_bp",120)
    return (hr>100 and (t>=38.3 or t<36.0)) or (sbp<90)

def _suspect_gi_bleed(vit: Dict[str,Any], s: List[str], labs: Dict[str,Any] | None) -> bool:
    sset = set(s)
    return any(x in sset for x in ["melena","hematemesis"]) and (_v(vit,"systolic_bp",999) < 100)

# -------- specialty bundles --------
SPECIALTY_RULES: Dict[str, List[Rule]] = {
    "cardiology": [
        ("Chest pain with hypotension", lambda vit, s, labs: (_v(vit,"systolic_bp",999) < 90) and any(x in s for x in ["chest pain","syncope","shortness of breath","dyspnea"])),
    ],
    "neurology": [
        ("New focal neuro deficit",     lambda vit, s, labs: any(x in s for x in ["weakness","facial droop","aphasia","vision loss"])),
        ("Syncope (evaluate for trauma and causes)", lambda vit, s, labs: "syncope" in s),
    ],
    "endocrinology": [
        ("Suspected DKA by labs/symptoms", lambda vit, s, labs: _suspect_dka(vit,s,labs)),
    ],
    "gastroenterology": [
        ("UGIB features (melena/hematemesis + hypotension)", lambda vit, s, labs: _suspect_gi_bleed(vit,s,labs)),
    ],
    "dermatology": [
        ("High fever with skin infection pattern",  lambda vit, s, labs: _v(vit,"temperature_c",0) > 39.5 and any(x in s for x in ["cellulitis","erysipelas","skin infection"])),
    ],
    "oncology": [
        ("Possible febrile neutropenia/sepsis",    lambda vit, s, labs: _suspect_sepsis(vit,s,labs)),
    ],
    "general practice/internal medicine": [
        ("Hypoxia or hypotension present",          lambda vit, s, labs: (_v(vit,"spo2",100) < 90) or (_v(vit,"systolic_bp",999) < 90)),
    ],
}

def _match_bundle(specialty: str) -> List[Rule]:
    spec = (specialty or "").strip().lower()
    for key, rules in SPECIALTY_RULES.items():
        if key in spec:
            return rules
    return []  # default none

def collect_red_flags(specialty: str, vitals: Dict[str, Any], symptoms: List[str], labs: Dict[str, Any] | None = None) -> List[str]:
    s = [x.lower() for x in (symptoms or [])]
    flags: List[str] = []

    # absolute → always apply
    for label, fn in ABSOLUTE_RULES:
        try:
            if fn(vitals, s, labs): flags.append(label)
        except Exception: pass

    # specialty bundle
    for label, fn in _match_bundle(specialty):
        try:
            if fn(vitals, s, labs): flags.append(label)
        except Exception: pass

    # lab-driven cross-specialty catches (optional universal checks)
    if _suspect_dka(vitals, s, labs): flags.append("Suspected DKA (labs/symptoms)")
    if _suspect_sepsis(vitals, s, labs): flags.append("Sepsis possible (vitals pattern)")

    return sorted(set(flags))

def naive_risk_bucket(vitals: Dict[str, Any], red_flags: List[str]) -> str:
    if red_flags: return "High"
    hr = vitals.get("heart_rate") or 0
    temp = vitals.get("temperature_c") or 0
    sbp = vitals.get("systolic_bp") or 999
    moderate = sum([
        hr > 110,
        38.5 < temp <= 39.5,
        90 <= sbp < 100,
    ])
    return "Moderate" if moderate >= 2 else "Low"
