# llm/llm_client.py
from typing import Literal, Optional
import os
import requests
import streamlit as st

Provider = Literal["gemini", "openai"]

class LLMClient:
    def __init__(self):
        cfg = st.secrets.get("api", {})
        self.provider: Provider = cfg.get("provider", "gemini")
        self.gemini_key = cfg.get("gemini_api_key") or os.getenv("GEMINI_API_KEY", "")
        self.openai_key = cfg.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")

    def generate(self, prompt: str, model: Optional[str] = None) -> str:
        if self.provider == "gemini":
            return self._gemini_generate(prompt, model or "gemini-1.5-flash")
        elif self.provider == "openai":
            return self._openai_generate(prompt, model or "gpt-4o-mini")
        return "[LLM Error: Unknown provider]"

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
                    # This is the KEY bit: tell Gemini to return JSON only
                    "response_mime_type": "application/json",
                },
            )

            resp = gmodel.generate_content(prompt)

            # 1) Prefer JSON text
            if getattr(resp, "text", None):
                return resp.text.strip()

            # 2) Try candidates/parts just in case
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
