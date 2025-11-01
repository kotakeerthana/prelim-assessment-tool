# utils/triage_rules.py
from typing import Dict, Any, List, Callable, Tuple, Optional

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
        # include English + Spanish variants so it works in either UI language
        ("Chest pain with hypotension",
         lambda vit, s, labs: (_v(vit, "systolic_bp", 999) < 90)
         and any(x in s for x in [
             "chest pain", "heart pain", "syncope",
             "shortness of breath", "dyspnea",
             "opresión torácica", "dolor torácico", "disnea"
         ])),
    ],
    "neurology": [
        ("New focal neuro deficit",
         lambda vit, s, labs: any(x in s for x in ["weakness", "facial droop", "aphasia", "vision loss"])),
        ("Syncope (evaluate for trauma and causes)",
         lambda vit, s, labs: "syncope" in s),
    ],
    "endocrinology": [
        ("Suspected DKA by labs/symptoms",
         lambda vit, s, labs: _suspect_dka(vit, s, labs)),
    ],
    "gastroenterology": [
        ("UGIB features (melena/hematemesis + hypotension)",
         lambda vit, s, labs: _suspect_gi_bleed(vit, s, labs)),
    ],
    "dermatology": [
        ("High fever with skin infection pattern",
         lambda vit, s, labs: _v(vit, "temperature_c", 0) > 39.5
         and any(x in s for x in ["cellulitis", "erysipelas", "skin infection"])),
    ],
    "oncology": [
        ("Possible febrile neutropenia/sepsis",
         lambda vit, s, labs: _suspect_sepsis(vit, s, labs)),
    ],
    # ⬇️ renamed to the code you use in the app (gp_im)
    "gp_im": [
        ("Hypoxia or hypotension present",
         lambda vit, s, labs: (_v(vit, "spo2", 100) < 90) or (_v(vit, "systolic_bp", 999) < 90)),
    ],
}

def _match_bundle(specialty: str) -> List[Rule]:
    """
    Match by exact code (e.g., 'cardiology', 'gp_im').
    The app now stores patient.specialty as a code, not a phrase.
    """
    spec = (specialty or "").strip().lower()
    return SPECIALTY_RULES.get(spec, [])

def collect_red_flags(specialty: str, vitals: Dict[str, Any], symptoms: List[str], labs: Dict[str, Any] | None = None) -> List[str]:
    s = [x.lower() for x in (symptoms or [])]
    flags: List[str] = []
    hr = (vitals or {}).get("heart_rate")
    rr = (vitals or {}).get("respiratory_rate")
    sbp = (vitals or {}).get("systolic_bp")
    spo2 = (vitals or {}).get("spo2")
    temp = (vitals or {}).get("temperature_c")

    if sbp is not None and sbp < 90: flags.append("Hipotensión (PAS < 90 mmHg)")
    if hr is not None and hr > 120: flags.append("Taquicardia marcada (FC > 120 lpm)")
    if rr is not None and rr > 30: flags.append("Taquipnea marcada (FR > 30/min)")
    if spo2 is not None and spo2 < 92: flags.append("Sat. O₂ baja (SpO₂ < 92%)")
    if temp is not None and temp >= 39.0: flags.append("Fiebre alta (T ≥ 39.0 °C)")
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
    temp = (vitals or {}).get("temperature_c")
    if temp is not None and temp >= 38.0:
        flags.append("Fever (≥ 38.0 °C)")

    return sorted(set(flags))

def naive_risk_bucket(entities, vitals, complaint: Optional[str] = None, severity: Optional[int] = None) -> str:
    """
    Conservative rule-based bucket:
    - HIGH: any hard instability; or ACS-pattern chest pain with severity >=8/10 or 'crushing'/radiating cues.
    - MODERATE: fever >=38.0 °C OR (>=37.8 °C with focal abdominal pain); or chest-pain-like + moderate abnormalities.
    - otherwise LOW.
    """
    bucket = "Low"

    sx = set([s.lower() for s in (entities or {}).get("symptoms", [])])
    complaint_txt = (complaint or "").lower()

    hr  = (vitals or {}).get("heart_rate")
    rr  = (vitals or {}).get("respiratory_rate")
    sbp = (vitals or {}).get("systolic_bp")
    spo2= (vitals or {}).get("spo2")
    temp= (vitals or {}).get("temperature_c")

    # Immediate HIGH: hard danger
    if (sbp is not None and sbp < 90) or (spo2 is not None and spo2 < 90):
        return "High"
    if (rr is not None and rr > 30) or (hr is not None and hr > 130) or (temp is not None and temp < 35.0):
        return "High"

    # ACS → HIGH even with normal vitals
    chest_like = any(k in sx or k in complaint_txt for k in [
        "chest pain", "heart pain", "opresión torácica", "dolor torácico",
        "dor torácica", "douleur thoracique", "胸痛"
    ])
    acs_cues = any(k in complaint_txt for k in [
        "crushing", "压榨", "opresivo", "oppressive", "radiating to left arm",
        "irrad", "diaphoresis", "sweating", "shortness of breath",
        "dyspnea", "disnea", "dispneia", "ضيق التنفس"
    ])
    if chest_like and ((severity or 0) >= 8 or acs_cues):
        return "High"

    # Fever → at least MODERATE
    if temp is not None and temp >= 38.0:
        return "Moderate"
    if (temp is not None and temp >= 37.8) and any(k in complaint_txt for k in [
        "rlq", "right lower quadrant", "lower right quadrant",
        "right iliac fossa", "append", "abdominal pain", "dolor abdominal",
        "dor abdominal", "douleur abdominale", "ألم بطني", "右下腹", "右下象限"
    ]):
        return "Moderate"

    # Chest pain + moderate abnormalities
    if chest_like and ((hr and hr > 100) or (rr and rr > 24) or (spo2 and spo2 < 92) or (temp and temp >= 38.0)):
        return "Moderate"

    return bucket
