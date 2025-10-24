from typing import List, Dict
import requests


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"




def search_pubmed(query: str, max_results: int = 3) -> List[Dict]:
    try:
        params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
        r = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=20)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        sid = ",".join(ids)
        sparams = {"db": "pubmed", "id": sid, "retmode": "json"}
        sr = requests.get(f"{EUTILS_BASE}/esummary.fcgi", params=sparams, timeout=20)
        sr.raise_for_status()
        j = sr.json()
        out = []
        for pid in ids:
            rec = j.get("result", {}).get(pid, {})
            title = rec.get("title", "")
            source = rec.get("source", "")
            pubdate = rec.get("pubdate", "")
            year = pubdate.split(" ")[0] if pubdate else ""
            out.append({"pmid": pid, "title": title, "journal": source, "year": year})
        return out
    except Exception:
        return []