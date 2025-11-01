import json
import re
import types

from utils.i18n import set_lang, language_name
from app import _translate_llm_json_if_needed

class FakeLLM:
    def __init__(self, translated_json):
        self.translated_json = translated_json
    def generate(self, prompt: str) -> str:
        # Return JSON string directly, as if model translated it
        return json.dumps(self.translated_json, ensure_ascii=False)

def test_translate_llm_json_into_spanish():
    set_lang("es")
    english_json = {
        "overview": "Presenting with chest pain.",
        "key_findings": "No key findings recorded.",
        "differentials": "Etiologies depend on location.",
        "risk_assessment": "Estimated Risk: Low.",
        "next_steps": "Clarify details.",
        "red_flags": "None elicited.",
        "limitations": "Automated first-level summary.",
        "references": ""
    }
    spanish_translated = {
        "overview": "Se presenta con dolor torácico.",
        "key_findings": "Sin hallazgos clave registrados.",
        "differentials": "Las etiologías dependen de la localización.",
        "risk_assessment": "Riesgo estimado: Bajo.",
        "next_steps": "Precisar detalles.",
        "red_flags": "No se identificaron.",
        "limitations": "Resumen automatizado de primer nivel.",
        "references": ""
    }
    llm = FakeLLM(spanish_translated)
    out = _translate_llm_json_if_needed(english_json, language_name(), llm)
    assert out["overview"].startswith("Se presenta")
    assert "Riesgo estimado" in out["risk_assessment"]
