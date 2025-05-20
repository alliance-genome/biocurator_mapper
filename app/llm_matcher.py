import json
from typing import List, Dict

import openai

from .config import OPENAI_API_KEY


class LLMMatcher:
    def __init__(self, openai_api_key: str = OPENAI_API_KEY):
        openai.api_key = openai_api_key

    def _build_prompt(self, passage: str, candidates: List[Dict]) -> str:
        candidate_descriptions = []
        for idx, c in enumerate(candidates, 1):
            desc = f"{idx}. {c['name']} ({c['id']}): {c.get('definition','') }"
            candidate_descriptions.append(desc)
        candidates_text = "\n".join(candidate_descriptions)
        prompt = (
            "You are an expert biomedical curator. Given the passage below and a list "
            "of candidate ontology terms, choose the single best matching term. "
            "Respond in JSON with keys 'id', 'name', 'confidence', and 'reason'.\n\n"
            f"Passage:\n{passage}\n\nCandidates:\n{candidates_text}\n"
        )
        return prompt

    def select_best_match(self, passage: str, candidates: List[Dict]) -> Dict:
        if not candidates:
            return {"error": "No candidates provided"}
        prompt = self._build_prompt(passage, candidates)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = response["choices"][0]["message"]["content"]
            result = json.loads(content)
            return result
        except Exception as e:  # broad catch for simplicity
            return {"error": str(e)}
