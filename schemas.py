from typing import Optional, List, Literal
from pydantic import BaseModel, Field

Specialty = Literal[
"Cardiology", "Oncology", "Neurology", "Endocrinology",
"Gastroenterology", "Dermatology", "General Practice/Internal Medicine", "Other"
]

class Vitals(BaseModel):
    heart_rate: Optional[int] = Field(None, ge=0, le=300)
    systolic_bp: Optional[int] = Field(None, ge=50, le=260)
    diastolic_bp: Optional[int] = Field(None, ge=30, le=160)
    respiratory_rate: Optional[int] = Field(None, ge=0, le=80)
    temperature_c: Optional[float] = Field(None, ge=30, le=45)
    spo2: Optional[int] = Field(None, ge=0, le=100)

class PatientInput(BaseModel):
    patient_id: Optional[str] = None
    specialty: Specialty


    # Demographics
    age: Optional[int] = Field(None, ge=0, le=120)
    sex: Optional[str] = Field(None)
    race_ethnicity: Optional[str] = None
    weight_kg: Optional[float] = Field(None, ge=0, le=500)
    height_cm: Optional[float] = Field(None, ge=0, le=260)
    bmi: Optional[float] = Field(None, ge=5, le=100)


    # Chief complaint
    complaint: Optional[str] = None
    duration: Optional[str] = None
    severity_1_10: Optional[int] = Field(None, ge=1, le=10)
    associated_symptoms: Optional[str] = None


    # History
    pmh: Optional[str] = None
    surgical_history: Optional[str] = None
    medications: Optional[str] = None
    allergies: Optional[str] = None
    family_history: Optional[str] = None


    # Social
    smoking_status: Optional[str] = None
    alcohol_consumption: Optional[str] = None
    illicit_drug_use: Optional[str] = None


    # Exam & tests
    vitals: Optional[Vitals] = None
    labs: Optional[str] = None
    imaging: Optional[str] = None
    other_tests: Optional[str] = None

class Entities(BaseModel):
    symptoms: List[str] = []
    conditions: List[str] = []
    medications: List[str] = []
    allergies: List[str] = []


class LLMRequest(BaseModel):
    system_prompt: str
    specialty: Specialty
    patient: PatientInput
    entities: Entities


class LLMResponse(BaseModel):
    overview: str
    key_findings: str
    differentials: str
    risk_assessment: str
    next_steps: str
    red_flags: str
    limitations: str
    references: str