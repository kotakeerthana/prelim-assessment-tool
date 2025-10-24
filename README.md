Preliminary Assessment Tool (MVP+)

 Non‑diagnostic, first‑level clinical summary tool for clinicians. Collects structured inputs, extracts entities, retrieves PubMed evidence, and generates a structured summary via LLM.

 ## Quickstart
 1) Create venv & install:
 ```bash
 python -m venv venv && source venv/bin/activate
 pip install -r requirements.txt
 ```
 2) Configure Streamlit Secrets (local `.streamlit/secrets.toml` or Streamlit Cloud Settings → Secrets):
 ```toml
 [api]
 provider = "gemini" # gemini | openai
 gemini_api_key = "YOUR_KEY"

 [storage]
 mode = "sqlite" # sqlite | csv
 db_path = "patient_logs.db"
 csv_path = "patient_logs.csv"
 ```
 3) Run:
 ```bash
 streamlit run app.py
 ```

 ## Notes
 - Educational demo — **not medical advice**. Always confirm with clinical judgment & guidelines.
 - Uses CPU‑only scispaCy; falls back to regex if model unavailable.
 - PubMed retrieval via NCBI E‑utilities (no key required for light use).
