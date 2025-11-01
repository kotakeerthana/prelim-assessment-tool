import json
from pathlib import Path

REQUIRED_KEYS = [
    "app.title", "nav.language",
    "report.title", "report.overview", "report.key_findings",
    "misc.estimated_risk", "vitals.label", "vitals.hr", "vitals.sbp", "vitals.dbp",
    "risk.low", "risk.moderate", "risk.high",
    "redflag.hypotension_sbp", "redflag.tachypnea_marked"
]

def load(path):
    p = Path(path)
    assert p.exists(), f"Missing i18n file: {path}"
    return json.loads(p.read_text(encoding="utf-8"))

def test_fr_pt_zh_ar_have_required_keys():
    for code in ["fr", "pt", "zh", "ar"]:
        data = load(f"i18n/{code}.json")
        for k in REQUIRED_KEYS:
            assert k in data, f"{code}.json missing key: {k}"
