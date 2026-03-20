from typing import Literal, Optional
import os
import requests
import streamlit as st
from utils.i18n import language_name

Provider = Literal["gemini", "openai"]


class LLMClient:
    def __init__(self):
        cfg = st.secrets.get("api", {})

        # Normalize provider so "Gemini" or " gemini " does not break logic
        self.provider: Provider = (cfg.get("provider", "gemini") or "gemini").strip().lower()  # type: ignore

        self.gemini_key = cfg.get("gemini_api_key") or os.getenv("GEMINI_API_KEY", "")
        self.openai_key = cfg.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")

        # IMPORTANT: default must be a model your key actually supports
        # Your key lists models like: models/gemini-pro-latest, models/gemini-flash-latest, models/gemini-2.5-pro, etc.
        self.gemini_model = cfg.get("model") or os.getenv("GEMINI_MODEL", "models/gemini-flash-latest")
        if self.gemini_model and not self.gemini_model.startswith("models/"):
            self.gemini_model = f"models/{self.gemini_model}"

        self.gemini_model = self._normalize_gemini_model(self.gemini_model)

        self.openai_model = cfg.get("openai_model") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def _normalize_gemini_model(self, model: str) -> str:
        """
        The google.generativeai SDK expects model names like 'models/gemini-pro-latest'.
        This normalizes common inputs so you never hit avoidable 404s.
        """
        m = (model or "").strip()
        if not m:
            return "models/gemini-pro-latest"
        if m.startswith("models/"):
            return m
        return f"models/{m}"

    def generate(self, prompt: str, model: Optional[str] = None) -> str:
        if self.provider == "gemini":
            chosen = model or self.gemini_model
            if chosen and not chosen.startswith("models/"):
                chosen = f"models/{chosen}"
            return self._gemini_generate(prompt, chosen)


        if self.provider == "openai":
            return self._openai_generate(prompt, model or self.openai_model)

        return "[LLM Error: Unknown provider]"

    def list_gemini_models(self) -> str:
        if not self.gemini_key:
            return "[LLM Error: Missing GEMINI_API_KEY]"
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)
            models = genai.list_models()
            lines = []
            for m in models:
                name = getattr(m, "name", "")
                methods = getattr(m, "supported_generation_methods", []) or []
                lines.append(f"{name} | {methods}")
            return "\n".join(lines) if lines else "(No models returned)"
        except Exception as e:
            return f"[LLM Error: {e}]"

    def _gemini_generate(self, prompt: str, model: str) -> str:
        if not self.gemini_key:
            return "[LLM Error: Missing GEMINI_API_KEY]"
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)

            gmodel = genai.GenerativeModel(
                model_name=model,
                generation_config={
                    "temperature": 0.2,
                    # Prefer JSON output (your report_generation expects JSON)
                    "response_mime_type": "application/json",
                },
            )

            resp = gmodel.generate_content(prompt)

            # 1) Prefer JSON text
            if getattr(resp, "text", None):
                return resp.text.strip()

            # 2) Fallback: candidates/parts
            try:
                parts = []
                for cand in getattr(resp, "candidates", []) or []:
                    content = getattr(cand, "content", None)
                    if content and getattr(content, "parts", None):
                        for p in content.parts:
                            t = getattr(p, "text", None)
                            if t:
                                parts.append(t)
                if parts:
                    return "\n".join(parts).strip()
            except Exception:
                pass

            return "[LLM Error: Empty response from Gemini]"

        except Exception as e:
            return f"[LLM Error: {e}]"

    def _openai_generate(self, prompt: str, model: str = "gpt-4o-mini") -> str:
        if not self.openai_key:
            return "[LLM Error: Missing OPENAI_API_KEY]"
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_key}",
                "Content-Type": "application/json",
            }
            data = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a careful clinical assistant generating a non-diagnostic first-level summary.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            }
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60,
            )
            r.raise_for_status()
            j = r.json()
            return j["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[LLM Error: {e}]"
