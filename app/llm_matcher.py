import json
import logging
from typing import List, Dict

import openai

from .config import OPENAI_API_KEY


class LLMMatcher:
    """Enhanced LLM matcher that leverages richer ontology term data for better matching."""
    
    def __init__(self, openai_api_key: str = OPENAI_API_KEY):
        openai.api_key = openai_api_key
        self.logger = logging.getLogger(__name__)

    def _build_prompt(self, passage: str, candidates: List[Dict]) -> str:
        """Build enhanced prompt with synonym and metadata information."""
        candidate_descriptions = []
        
        for idx, c in enumerate(candidates, 1):
            # Build rich description including synonyms
            desc_parts = [f"{idx}. {c['name']} ({c['id']})"]
            
            # Add namespace info
            if c.get('namespace'):
                desc_parts.append(f"[{c['namespace']}]")
            
            # Add definition
            definition = c.get('definition', '')
            if definition:
                desc_parts.append(f"Definition: {definition}")
            
            # Add synonyms for better context
            exact_syns = c.get('exact_synonyms', [])
            if exact_syns:
                desc_parts.append(f"Exact synonyms: {', '.join(exact_syns[:3])}")
            
            narrow_syns = c.get('narrow_synonyms', [])
            if narrow_syns:
                desc_parts.append(f"Narrow synonyms: {', '.join(narrow_syns[:3])}")
            
            # Add similarity score if available
            if 'similarity_certainty' in c:
                certainty = c['similarity_certainty']
                desc_parts.append(f"Similarity: {certainty:.3f}")
            
            candidate_descriptions.append(" | ".join(desc_parts))
        
        candidates_text = "\n".join(candidate_descriptions)
        
        prompt = (
            "You are an expert biomedical curator with deep knowledge of Gene Ontology terms. "
            "Given a scientific passage and candidate ontology terms (with their synonyms and metadata), "
            "choose the single best matching term.\n\n"
            
            "Consider:\n"
            "- Semantic similarity between passage and term definition\n"
            "- Relevance of synonyms to the passage content\n"
            "- Appropriateness of the ontology namespace (biological_process, molecular_function, cellular_component)\n"
            "- Vector similarity scores as a guide\n\n"
            
            "Respond ONLY in valid JSON format with these exact keys:\n"
            "{\n"
            '  "id": "GO:XXXXXXX",  // Must be one of the provided candidate IDs\n'
            '  "name": "exact term name",\n'
            '  "confidence": 0.95,  // Float between 0-1\n'
            '  "reason": "brief explanation focusing on why this term best matches the passage"\n'
            "}\n\n"
            
            f"Scientific passage:\n{passage}\n\n"
            f"Candidate terms:\n{candidates_text}\n"
        )
        return prompt

    async def select_best_match(self, passage: str, candidates: List[Dict]) -> Dict:
        """Enhanced matching using richer term data and improved prompting."""
        if not candidates:
            return {"error": "No candidates provided"}
        
        prompt = self._build_prompt(passage, candidates)
        
        try:
            self.logger.info(f"Calling OpenAI for enhanced matching of passage: '{passage[:50]}...'")
            
            response = await openai.chat.completions.create(
                model="gpt-4",  # Use GPT-4 for better reasoning
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=500,   # Sufficient for structured response
            )
            
            content = response.choices[0].message.content.strip()
            
            # Handle potential markdown code blocks
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            
            # Validate result structure
            required_keys = {"id", "name", "confidence", "reason"}
            if not all(key in result for key in required_keys):
                missing_keys = required_keys - set(result.keys())
                return {"error": f"LLM response missing required keys: {missing_keys}"}
            
            # Validate that chosen ID is from candidates
            candidate_ids = {c["id"] for c in candidates}
            if result["id"] not in candidate_ids:
                return {"error": f"LLM chose invalid ID {result['id']} not in candidates"}
            
            # Add the full term data for the selected match
            for candidate in candidates:
                if candidate["id"] == result["id"]:
                    result.update({
                        "definition": candidate.get("definition", ""),
                        "exact_synonyms": candidate.get("exact_synonyms", []),
                        "namespace": candidate.get("namespace", ""),
                        "similarity_certainty": candidate.get("similarity_certainty", 0.0)
                    })
                    break
            
            self.logger.info(f"Selected match: {result['id']} with confidence {result['confidence']}")
            return result
            
        except openai.OpenAIError as e:
            self.logger.exception("OpenAI API error in enhanced matching")
            return {"error": f"OpenAI API error: {e}"}
        except json.JSONDecodeError as e:
            self.logger.exception(f"Failed to parse OpenAI response: {content}")
            return {"error": f"JSON decode error: {e}"}
        except Exception as e:
            self.logger.exception("Unexpected error during enhanced LLM matching")
            return {"error": str(e)}

    async def explain_match(self, passage: str, selected_term: Dict) -> str:
        """Generate detailed explanation of why a term was selected."""
        explanation_prompt = (
            f"Explain in 2-3 sentences why the Gene Ontology term "
            f"'{selected_term['name']}' ({selected_term['id']}) is the best match "
            f"for the scientific passage: '{passage}'\n\n"
            f"Term definition: {selected_term.get('definition', '')}\n"
            f"Term synonyms: {', '.join(selected_term.get('exact_synonyms', []))}"
        )
        
        try:
            response = await openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": explanation_prompt}],
                temperature=0.3,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.exception("Failed to generate explanation")
            return f"Match selected based on semantic similarity (explanation generation failed: {e})"