import pytest

from utils.triage_rules import naive_risk_bucket, collect_red_flags

def test_chest_pain_crushing_severity_10_is_high():
    entities = {"symptoms": ["chest pain"]}
    vitals = {"heart_rate": 80, "respiratory_rate": 16, "systolic_bp": 120, "spo2": 98, "temperature_c": 36.8}
    complaint = "Crushing chest pain radiating to left arm"
    out = naive_risk_bucket(entities, vitals, complaint=complaint, severity=10)
    assert out == "High"

def test_rlq_abdominal_pain_with_37_8_is_moderate():
    entities = {"symptoms": ["abdominal pain"]}
    vitals = {"temperature_c": 37.8, "heart_rate": 92, "respiratory_rate": 18, "systolic_bp": 118, "spo2": 98}
    complaint = "Lower right quadrant abdominal pain"
    out = naive_risk_bucket(entities, vitals, complaint=complaint, severity=3)
    assert out == "Moderate"

def test_hard_instability_rules_force_high():
    entities = {"symptoms": []}
    vitals = {"systolic_bp": 85, "spo2": 89}
    out = naive_risk_bucket(entities, vitals, complaint=None, severity=5)
    assert out == "High"

def test_collect_red_flags_includes_fever_when_38_or_more():
    entities = {"symptoms": []}
    vitals = {"temperature_c": 38.1, "heart_rate": 95, "systolic_bp": 118}
    flags = collect_red_flags("General Practice/Internal Medicine", vitals, entities.get("symptoms", []), labs=None)
    # Depending on where you added fever flag, assert any one that should appear:
    assert any(("Fever" in f) or ("Fiebre" in f) or ("Fièvre" in f) or ("发热" in f) for f in flags)
