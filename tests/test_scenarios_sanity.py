import pytest
from utils.triage_rules import naive_risk_bucket, collect_red_flags

def rb(entities, vitals, complaint, severity):
    return naive_risk_bucket(entities, vitals, complaint=complaint, severity=severity)

def test_low_msk_chest_wall():
    e = {"symptoms": ["chest pain"]}
    v = {"heart_rate": 78, "respiratory_rate": 14, "systolic_bp": 122, "spo2": 99, "temperature_c": 36.8}
    out = rb(e, v, "Chest pain reproducible with palpation", 3)
    assert out == "Low"

def test_moderate_fever_rlq():
    e = {"symptoms": ["abdominal pain"]}
    v = {"temperature_c": 38.0, "heart_rate": 92, "respiratory_rate": 18, "systolic_bp": 118, "spo2": 98}
    out = rb(e, v, "Lower right quadrant abdominal pain", 3)
    assert out == "Moderate"

def test_high_crushing_cp_10of10_normal_vitals():
    e = {"symptoms": ["chest pain"]}
    v = {"heart_rate": 82, "respiratory_rate": 16, "systolic_bp": 128, "spo2": 98, "temperature_c": 36.8}
    out = rb(e, v, "Crushing chest pain radiating to left arm", 10)
    assert out == "High"

def test_high_hypoxia():
    e = {"symptoms": ["shortness of breath"]}
    v = {"spo2": 88, "heart_rate": 104, "respiratory_rate": 22, "systolic_bp": 126, "temperature_c": 36.7}
    out = rb(e, v, "Shortness of breath", 5)
    assert out == "High"

def test_high_tachypnea():
    e = {"symptoms": ["shortness of breath"]}
    v = {"respiratory_rate": 34, "heart_rate": 96, "systolic_bp": 118, "spo2": 97, "temperature_c": 36.8}
    out = rb(e, v, "Shortness of breath", 5)
    assert out == "High"

def test_high_sepsis_pattern():
    e = {"symptoms": ["fever", "chills"]}
    v = {"heart_rate": 112, "temperature_c": 38.5, "systolic_bp": 88, "respiratory_rate": 22, "spo2": 95}
    out = rb(e, v, "Fever, chills", 5)
    assert out == "High"

def test_high_hypothermia():
    e = {"symptoms": ["lethargy"]}
    v = {"temperature_c": 34.5, "heart_rate": 88, "respiratory_rate": 16, "systolic_bp": 122, "spo2": 98}
    out = rb(e, v, "Lethargy", 5)
    assert out == "High"

def test_high_tachycardia():
    e = {"symptoms": ["palpitations"]}
    v = {"heart_rate": 145, "respiratory_rate": 18, "systolic_bp": 122, "spo2": 98, "temperature_c": 36.8}
    out = rb(e, v, "Palpitations", 6)
    assert out == "High"
