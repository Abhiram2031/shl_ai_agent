import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

CATALOG_PATH = Path("data/shl_catalog.json")

NAME_ALIASES = {
    "opq": "Occupational Personality Questionnaire OPQ32r",
    "opq32r": "Occupational Personality Questionnaire OPQ32r",
    "gsa": "Global Skills Assessment",
    "dsi": "Dependability and Safety Instrument (DSI)",
    "verify g+": "SHL Verify Interactive G+",
    "verify interactive g+": "SHL Verify Interactive G+",
}


def _load_catalog() -> list[dict]:
    if not CATALOG_PATH.exists():
        return []
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []


def _make_document(item: dict) -> str:
    """
    Flatten all searchable fields into one string for TF-IDF indexing.
    """
    keys = item.get("keys", [])
    keys_str = " ".join(keys) if isinstance(keys, list) else ""
    job_levels = item.get("job_levels", [])
    levels_str = " ".join(job_levels) if isinstance(job_levels, list) else ""

    parts = [
        item.get("name", ""),
        item.get("description", ""),
        keys_str,
        levels_str,
    ]
    return " ".join(p for p in parts if p).lower()


def _get_url(item: dict) -> str:
    """Catalog has either 'url' or 'link' (handles both)"""
    return item.get("url") or item.get("link") or ""


def _get_test_type(item: dict) -> str:
    """
    Map the first entry in 'keys' to a standard SHL test type code.
    
    a short code (e.g., "K", "P") in the response's
    'test_type' field. Returning the full key string or an empty value would
    violate the schema and cause the submission to fail.
    """
    keys = item.get("keys", [])
    if not keys:
        return ""
    first_key = keys[0]
    mapping = {
        "Knowledge & Skills": "K",
        "Personality & Behavior": "P",
        "Ability & Aptitude": "A",
        "Simulations": "S",
        "Competencies": "C",
        "Biodata & Situational Judgment": "B",
        "Assessment Exercises": "E",
        "Development & 360": "D",
    }
    return mapping.get(first_key, "G")   # G = General


def _score_boost(item: dict, query: str) -> float:
    """Small intent boosts for common SHL core instruments."""
    name = item.get("name", "").lower()
    query_lower = query.lower()
    boost = 0.0

    if "personality" in query_lower and name == "occupational personality questionnaire opq32r":
        boost += 0.08
    if any(term in query_lower for term in ("cognitive", "reasoning", "ability")) and name == "shl verify interactive g+":
        boost += 0.08
    if "graduate" in query_lower and any(term in query_lower for term in ("situational", "judgement", "judgment", "scenario")) and name == "graduate scenarios":
        boost += 0.08
    if "leadership" in query_lower and name in {
        "occupational personality questionnaire opq32r",
        "opq leadership report",
        "opq universal competency report 2.0",
    }:
        boost += 0.20
    if any(term in query_lower for term in ("engineer", "developer", "coding", "programming")):
        if name == "smart interview live coding":
            boost += 0.20
        if "senior" in query_lower and name in {
            "shl verify interactive g+",
            "occupational personality questionnaire opq32r",
        }:
            boost += 0.08
    if "rust" in query_lower or "networking" in query_lower or "infrastructure" in query_lower:
        if name in {"linux programming (general)", "networking and implementation (new)"}:
            boost += 0.15
    if any(term in query_lower for term in ("contact centre", "contact center", "inbound calls", "call center")):
        if name in {
            "svar - spoken english (us) (new)",
            "contact center call simulation (new)",
            "entry level customer serv-retail & contact center",
            "customer service phone simulation",
        }:
            boost += 0.22
    if "financial analyst" in query_lower or "finance" in query_lower:
        if name in {
            "shl verify interactive - numerical reasoning",
            "financial accounting (new)",
            "basic statistics (new)",
            "graduate scenarios",
        }:
            boost += 0.16
        if name == "occupational personality questionnaire opq32r":
            boost += 0.08
    if "sales" in query_lower and any(term in query_lower for term in ("audit", "reskill", "re-skill", "organization")):
        if name in {
            "global skills assessment",
            "global skills development report",
            "occupational personality questionnaire opq32r",
            "opq mq sales report",
            "sales transformation 2.0 - individual contributor",
        }:
            boost += 0.18
    if any(term in query_lower for term in ("safety", "plant operator", "chemical facility", "procedure compliance")):
        if name in {
            "dependability and safety instrument (dsi)",
            "manufac. & indust. - safety & dependability 8.0",
            "workplace health and safety (new)",
        }:
            boost += 0.12
    if any(term in query_lower for term in ("hipaa", "healthcare", "patient records")):
        if name in {
            "hipaa (security)",
            "medical terminology (new)",
            "microsoft word 365 - essentials (new)",
            "dependability and safety instrument (dsi)",
            "occupational personality questionnaire opq32r",
        }:
            boost += 0.11
    if any(term in query_lower for term in ("admin assistant", "excel", "word daily")):
        if name in {
            "ms excel (new)",
            "ms word (new)",
            "microsoft excel 365 (new)",
            "microsoft word 365 (new)",
            "occupational personality questionnaire opq32r",
        }:
            boost += 0.10

    return boost


class CatalogRetriever:
    def __init__(self):
        self.items = _load_catalog()
        self._docs = [_make_document(item) for item in self.items]
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams: "java developer" scores as a unit
            min_df=1,
            stop_words="english",
        )
        # Only fit if documents are present
        if self._docs:
            self._matrix = self._vectorizer.fit_transform(self._docs)
        else:
            self._matrix = None

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        TF-IDF cosine similarity search.
        Returns up to top_k items with a '_score' field appended.
        Falls back to empty list if catalog is empty.
        """
        if self._matrix is None or not query.strip():
            return []

        q_vec = self._vectorizer.transform([query.lower()])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        boosts = np.array([_score_boost(item, query) for item in self.items])
        scores = scores + boosts
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for idx in ranked_indices:
            # if scores[idx] < 0.01:   # Removed the 0.01 threshold — at 35 items we need everything ranked, even low scorers, so the LLM can reason across the full catalog
            #     break
            # item = self.items[idx]
            results.append({
                **self.items[idx],
                "url": _get_url(self.items[idx]),
                "test_type": _get_test_type(self.items[idx]),
                "_score": float(scores[idx]),
            })
            if len(results) >= top_k:
                break

        return results

    def normalize_item(self, item: dict) -> dict:
        """Return an item with normalized URL and test type fields."""
        return {
            **item,
            "url": _get_url(item),
            "test_type": item.get("test_type") or _get_test_type(item),
        }

    def get_by_name(self, name: str) -> dict | None:
        """Exact then fuzzy name match — used for compare queries."""
        name_lower = name.lower().strip()
        alias_target = NAME_ALIASES.get(name_lower)
        if alias_target:
            for item in self.items:
                if item.get("name", "").lower() == alias_target.lower():
                    return self.normalize_item(item)
        for item in self.items:
            if item.get("name", "").lower() == name_lower:
                return self.normalize_item(item)
        for item in self.items:
            if name_lower in item.get("name", "").lower():
                return self.normalize_item(item)
        return None

    def format_for_prompt(self, items: list[dict]) -> str:
        """
        Compact single-line format per item to stay under Groq's token limit.
        Only name, test_type, keys, and url — the LLM needs nothing else.
        """
        lines = []
        for i, item in enumerate(items, 1):
            keys = item.get("keys", [])
            keys_str = "|".join(keys[:2]) if isinstance(keys, list) else ""
            test_type = item.get("test_type") or _get_test_type(item)
            url = _get_url(item)
            lines.append(
                f"{i}. {item.get('name')}|{test_type}|{keys_str}|{url}"
            )
        return "\n".join(lines)


    def valid_urls(self) -> set[str]:
        """Set of all known catalog URLs(used for hallucination whitelist.)"""
        return {_get_url(item) for item in self.items if _get_url(item)}

    def items_by_url(self) -> dict[str, dict]:
        return {
            _get_url(item): self.normalize_item(item)
            for item in self.items
            if _get_url(item)
        }

    def all_names(self) -> list[str]:
        return [item.get("name", "") for item in self.items]
