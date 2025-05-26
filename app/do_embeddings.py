import logging
import json
from typing import List, Dict, Any, Optional, Tuple
import yaml
from pathlib import Path
from functools import lru_cache

from .openai_client import get_embedding_client, EmbeddingResult, OpenAIClientError
from .models import DOTerm
from .config import Config

logger = logging.getLogger(__name__)


class DOEmbeddingGenerator:
    """Generate embeddings for Disease Ontology terms."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize DO Embedding Generator.
        
        Args:
            config: Configuration object. If None, creates a new one.
        """
        self.config = config or Config()
        self.embeddings_config = self.config.get_embeddings_config()
        
        # Initialize OpenAI embedding client
        self.client = get_embedding_client(config=self.config)
        
        # Get DO-specific configuration
        self.do_config = self.embeddings_config.get('do_specific', {})
        self.processing_config = self.embeddings_config.get('processing', {})
        self.vectorize_fields = self.embeddings_config.get('vectorize_fields', {})
    
    def _meets_quality_filters(self, term: DOTerm) -> bool:
        """Check if term meets quality filters.
        
        Args:
            term: DO term to check.
            
        Returns:
            True if term meets quality requirements.
        """
        quality_filters = self.do_config.get('quality_filters', {})
        
        # Check minimum definition length
        min_def_length = quality_filters.get('min_definition_length', 0)
        if min_def_length > 0:
            if not term.definition or len(term.definition) < min_def_length:
                return False
                
        # Check exclude patterns
        exclude_patterns = quality_filters.get('exclude_patterns', [])
        for pattern in exclude_patterns:
            if pattern.lower() in term.name.lower():
                return False
            if term.definition and pattern.lower() in term.definition.lower():
                return False
                
        # Check if definition is required
        metadata_config = self.do_config.get('include_metadata', {})
        if metadata_config.get('definition_required', False):
            if not term.definition or not term.definition.strip():
                return False
                
        # Check if obsolete terms should be excluded
        if not metadata_config.get('include_obsolete', True):
            if hasattr(term, 'is_obsolete') and term.is_obsolete:
                return False
                
        return True
    
    def _preprocess_term(self, term: DOTerm) -> str:
        """Preprocess DO term for embedding generation.
        
        Args:
            term: DO term to preprocess.
        
        Returns:
            Preprocessed text for embedding.
        """
        text_composition = self.do_config.get('text_composition', {})
        separator = text_composition.get('separator', ' | ')
        max_length = text_composition.get('max_text_length', 8000)
        
        # Build text parts with weights
        text_parts = []
        
        # Primary text (usually the name)
        primary_field = text_composition.get('primary_text', 'name')
        if primary_field == 'name' and term.name:
            weight = self.vectorize_fields.get('name', 1.0)
            if weight > 0:
                text_parts.append(f"Name: {term.name}")
        
        # Context fields
        context_fields = text_composition.get('context_fields', ['definition', 'synonyms'])
        
        if 'definition' in context_fields and term.definition:
            weight = self.vectorize_fields.get('definition', 0.8)
            if weight > 0:
                text_parts.append(f"Definition: {term.definition}")
        
        if 'synonyms' in context_fields and term.synonyms:
            weight = self.vectorize_fields.get('synonyms', 0.6)
            if weight > 0:
                synonym_types = self.do_config.get('synonym_types', {})
                
                # Process synonyms by type with weights
                weighted_synonyms = []
                for syn_type, syns in term.synonyms.items():
                    type_weight = synonym_types.get(syn_type, 0.5)
                    if type_weight > 0 and syns:
                        for syn in syns:
                            weighted_synonyms.append(f"{syn_type}: {syn}")
                
                if weighted_synonyms:
                    text_parts.append(f"Synonyms: {', '.join(weighted_synonyms)}")
        
        # Cross-references with filtering
        if 'xrefs' in context_fields and hasattr(term, 'xrefs') and term.xrefs:
            weight = self.vectorize_fields.get('xrefs', 0.4)
            if weight > 0:
                metadata_config = self.do_config.get('include_metadata', {})
                prioritized_sources = metadata_config.get('xref_sources', [])
                
                # Filter and prioritize xrefs
                filtered_xrefs = []
                for xref in term.xrefs:
                    for source in prioritized_sources:
                        if xref.startswith(source):
                            filtered_xrefs.append(xref)
                            break
                
                if filtered_xrefs:
                    text_parts.append(f"References: {', '.join(filtered_xrefs[:5])}")  # Limit to 5
        
        # Combine parts
        combined_text = separator.join(text_parts)
        
        # Truncate if too long
        if len(combined_text) > max_length:
            combined_text = combined_text[:max_length-3] + "..."
            
        return combined_text
    
    def filter_terms(self, terms: List[DOTerm]) -> Tuple[List[DOTerm], List[str]]:
        """Filter terms based on quality criteria.
        
        Args:
            terms: List of DO terms to filter.
            
        Returns:
            Tuple of (filtered_terms, rejection_reasons).
        """
        filtered_terms = []
        rejection_reasons = []
        
        for term in terms:
            if self._meets_quality_filters(term):
                filtered_terms.append(term)
            else:
                reason = f"Term {term.id} ({term.name}) rejected by quality filters"
                rejection_reasons.append(reason)
                logger.debug(reason)
                
        logger.info(f"Filtered {len(terms)} terms to {len(filtered_terms)} valid terms")
        return filtered_terms, rejection_reasons
    
    def generate_embeddings(
        self, 
        terms: List[DOTerm], 
        batch_size: Optional[int] = None,
        apply_filters: bool = True
    ) -> Tuple[List[EmbeddingResult], List[str]]:
        """Generate embeddings for a list of DO terms.
        
        Args:
            terms: List of DO terms to embed.
            batch_size: Number of terms to process in each batch.
            apply_filters: Whether to apply quality filters.
        
        Returns:
            Tuple of (embedding_results, rejection_reasons).
            
        Raises:
            OpenAIClientError: If embedding generation fails.
        """
        if not terms:
            return [], []
            
        # Filter terms if requested
        if apply_filters:
            filtered_terms, rejection_reasons = self.filter_terms(terms)
        else:
            filtered_terms = terms
            rejection_reasons = []
            
        if not filtered_terms:
            logger.warning("No valid terms to process after filtering")
            return [], rejection_reasons
        
        # Use batch size from config if not specified
        if batch_size is None:
            batch_size = self.processing_config.get('batch_size', 100)
        
        # Process terms in batches
        all_results = []
        total_batches = (len(filtered_terms) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(filtered_terms), batch_size):
            batch_terms = filtered_terms[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            
            try:
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_terms)} terms)")
                
                # Preprocess terms
                preprocessed_texts = []
                valid_terms = []
                
                for term in batch_terms:
                    try:
                        text = self._preprocess_term(term)
                        if text.strip():  # Only include non-empty texts
                            preprocessed_texts.append(text)
                            valid_terms.append(term)
                        else:
                            logger.warning(f"Empty text for term {term.id}")
                    except Exception as e:
                        logger.error(f"Error preprocessing term {term.id}: {e}")
                        continue
                
                if not preprocessed_texts:
                    logger.warning(f"No valid texts in batch {batch_num}")
                    continue
                
                # Generate embeddings
                batch_results = self.client.generate_embeddings(preprocessed_texts)
                
                # Associate embeddings with terms
                for i, embedding_result in enumerate(batch_results):
                    if i < len(valid_terms):
                        # Add term metadata to embedding result
                        enhanced_result = EmbeddingResult(
                            embedding=embedding_result.embedding,
                            text=embedding_result.text,
                            token_count=embedding_result.token_count,
                            model=embedding_result.model
                        )
                        # Store term ID for reference
                        enhanced_result.term_id = valid_terms[i].id
                        enhanced_result.term_name = valid_terms[i].name
                        all_results.append(enhanced_result)
                
                logger.info(f"Batch {batch_num} completed: {len(batch_results)} embeddings generated")
                
            except OpenAIClientError as e:
                logger.error(f"OpenAI error in batch {batch_num}: {e}")
                if not self.processing_config.get('retry_failed', True):
                    raise
                continue
            except Exception as e:
                logger.error(f"Unexpected error in batch {batch_num}: {e}")
                raise
        
        logger.info(f"Embedding generation completed: {len(all_results)} embeddings from {len(filtered_terms)} terms")
        return all_results, rejection_reasons
    
    def generate_single_embedding(self, term: DOTerm) -> Optional[EmbeddingResult]:
        """Generate embedding for a single DO term.
        
        Args:
            term: DO term to embed.
            
        Returns:
            EmbeddingResult or None if term doesn't meet quality filters.
        """
        if not self._meets_quality_filters(term):
            logger.debug(f"Term {term.id} rejected by quality filters")
            return None
            
        try:
            text = self._preprocess_term(term)
            if not text.strip():
                logger.warning(f"Empty text for term {term.id}")
                return None
                
            result = self.client.generate_embedding(text)
            
            # Add term metadata
            result.term_id = term.id
            result.term_name = term.name
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating embedding for term {term.id}: {e}")
            raise