from typing import Dict, Any
import streamlit as st
import pandas as pd
import os
import datetime as dt
from sqlalchemy import create_engine


SCHEMA_COLUMNS = [
    "timestamp", "patient_id", "specialty", "age", "sex", "race_ethnicity", "weight", "height", "bmi",
    "complaint", "duration", "severity", "assoc_symptoms",
    "pmh", "surg_hx", "meds", "allergies", "fam_hx",
    "smoking", "alcohol", "drugs",
    "vitals", "labs", "imaging", "other_tests",
    "entities", "pubmed_refs", "report_raw", "report_rendered"
]


def _now_iso():
    return dt.datetime.utcnow().isoformat()




def _storage_mode():
    return st.secrets.get("storage", {}).get("mode", "sqlite")




def log_record(record: Dict[str, Any]):
    mode = _storage_mode()
    record = {k: record.get(k) for k in SCHEMA_COLUMNS}
    if mode == "csv":
        path = st.secrets.get("storage", {}).get("csv_path", "patient_logs.csv")
        df = pd.DataFrame([record])
        header = not os.path.exists(path)
        df.to_csv(path, mode='a', index=False, header=header)
    else:
        path = st.secrets.get("storage", {}).get("db_path", "patient_logs.db")
        engine = create_engine(f"sqlite:///{path}")
        df = pd.DataFrame([record])
        df.to_sql("patient", con=engine, if_exists="append", index=False)