"""Vector Store Manager - Handles semantic search operations."""

from typing import Dict, List, Any, Optional
import asyncio
import re
from sentence_transformers import SentenceTransformer

from core.database import vector_store, db_manager
from agents.router import DataSource
from config.settings import settings
import structlog

logger = structlog.get_logger()


class VectorManager:
    """Manages vector search operations across different embedding models."""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.default_table = "knowledge_embeddings"
        
        # Common question words to filter out for better semantic matching
        self.question_words = {
            'do', 'you', 'believe', 'think', 'what', 'why', 'how', 'when', 'where', 
            'who', 'which', 'would', 'could', 'should', 'can', 'will', 'is', 'are',
            'was', 'were', 'have', 'has', 'had', 'does', 'did', 'there', 'something',
            'anything', 'nothing', 'everything', 'more', 'less', 'most', 'least'
        }
    
    async def initialize(self):
        """Initialize vector store tables."""
        await vector_store.create_embedding_table(self.default_table)
        logger.info("Vector store initialized")
    
    async def query(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute vector similarity search with enhanced query processing."""
        try:
            query_text = query_params.get("query_text", "")
            limit = query_params.get("limit", settings.max_results)
            threshold = query_params.get("threshold", settings.confidence_threshold)
            
            # Enhanced query processing for better semantic matching
            processed_queries = self._process_query_for_search(query_text)
            
            all_results = []
            
            # Search with multiple query variations
            for processed_query in processed_queries:
                query_embedding = self.embedding_model.encode(processed_query).tolist()
                
                # Perform similarity search with lower threshold initially
                search_results = await vector_store.similarity_search(
                    self.default_table,
                    query_embedding,
                    limit=limit * 2,  # Get more results initially
                    threshold=max(0.3, threshold - 0.2)  # Lower threshold for broader search
                )
                
                # Add query variation info to results
                for result in search_results:
                    result['query_variation'] = processed_query
                    result['original_query'] = query_text
                
                all_results.extend(search_results)
            
            # Remove duplicates and re-rank results
            unique_results = self._deduplicate_and_rerank(all_results, query_text, threshold)
            
            # Limit to requested number of results
            final_results = unique_results[:limit]
            
            logger.info("Enhanced vector search completed",
                       original_query=query_text,
                       query_variations=len(processed_queries),
                       total_found=len(all_results),
                       after_dedup=len(unique_results),
                       final_results=len(final_results))
            
            return {
                "source": DataSource.VECTOR_STORE.value,
                "results": final_results,
                "query_embedding_dim": len(query_embedding),
                "total_results": len(final_results),
                "query_variations_used": len(processed_queries)
            }
            
        except Exception as e:
            logger.error("Vector search failed", error=str(e))
            return {"source": DataSource.VECTOR_STORE.value, "results": [], "error": str(e)}
    
    def _process_query_for_search(self, query_text: str) -> List[str]:
        """Process query to create multiple search variations for better matching."""
        processed_queries = []
        
        # Original query
        processed_queries.append(query_text)
        
        # Extract key terms by removing question words and common phrases
        key_terms = self._extract_key_terms(query_text)
        if key_terms and key_terms != query_text:
            processed_queries.append(key_terms)
        
        # Extract noun phrases and important concepts
        concepts = self._extract_concepts(query_text)
        for concept in concepts:
            if concept not in processed_queries:
                processed_queries.append(concept)
        
        # Create focused query from main topic
        focused_query = self._create_focused_query(query_text)
        if focused_query and focused_query not in processed_queries:
            processed_queries.append(focused_query)
        
        # Extract brand names and create brand-focused queries
        brand_queries = self._extract_brand_queries(query_text)
        for brand_query in brand_queries:
            if brand_query not in processed_queries:
                processed_queries.append(brand_query)
        
        # Create title-style variations (with dashes and formal formatting)
        title_variations = self._create_title_variations(query_text)
        for variation in title_variations:
            if variation not in processed_queries:
                processed_queries.append(variation)
        
        return processed_queries[:6]  # Increased to 6 variations for better coverage
    
    def _extract_key_terms(self, query_text: str) -> str:
        """Extract key terms by removing question words and common phrases."""
        # Convert to lowercase and split into words
        words = re.findall(r'\b\w+\b', query_text.lower())
        
        # Remove question words and common terms
        key_words = [word for word in words if word not in self.question_words and len(word) > 2]
        
        return ' '.join(key_words)
    
    def _extract_concepts(self, query_text: str) -> List[str]:
        """Extract important concepts and noun phrases."""
        concepts = []
        
        # Look for quoted phrases or capitalized terms
        quoted_phrases = re.findall(r'"([^"]*)"', query_text)
        concepts.extend(quoted_phrases)
        
        # Look for capitalized terms (proper nouns)
        capitalized_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query_text)
        concepts.extend(capitalized_terms)
        
        # Look for specific patterns like "X Triangle", "X explanation", etc.
        patterns = [
            r'(\w+\s+triangle)',
            r'(\w+\s+explanation)',
            r'(\w+\s+theory)',
            r'(\w+\s+phenomenon)',
            r'(\w+\s+mystery)',
            r'(\w+\s+disappearances?)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query_text, re.IGNORECASE)
            concepts.extend(matches)
        
        return [concept.strip() for concept in concepts if len(concept.strip()) > 3]
    
    def _extract_brand_queries(self, query_text: str) -> List[str]:
        """Extract brand-focused queries from the original query."""
        brand_queries = []
        
        # Look for potential brand names (capitalized words)
        import re
        brand_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Multi-word capitalized terms
        ]
        
        brands = []
        for pattern in brand_patterns:
            matches = re.findall(pattern, query_text)
            brands.extend(matches)
        
        # Create brand-focused queries
        for brand in brands:
            if len(brand) > 3:
                # Brand + key terms from query
                key_terms = self._extract_key_terms(query_text).split()
                for term in key_terms:
                    if term.lower() not in brand.lower():
                        brand_query = f"{brand} {term}"
                        brand_queries.append(brand_query)
                
                # Just the brand name
                brand_queries.append(brand)
        
        return brand_queries
    
    def _create_title_variations(self, query_text: str) -> List[str]:
        """Create title-style variations that might match document titles."""
        variations = []
        
        # Extract key components
        import re
        words = re.findall(r'\b\w+\b', query_text)
        
        # Filter out question words and common terms
        meaningful_words = []
        for word in words:
            if (word.lower() not in self.question_words and 
                len(word) > 2 and 
                not word.lower() in ['about', 'the']):
                meaningful_words.append(word)
        
        if len(meaningful_words) >= 2:
            # Create title-case version
            title_case = ' '.join(word.capitalize() for word in meaningful_words)
            variations.append(title_case)
            
            # Create dash-separated version (common in titles)
            dash_version = ' - '.join(word.capitalize() for word in meaningful_words)
            variations.append(dash_version)
            
            # Create variations with common title words
            if any(word.lower() in ['wine', 'gift', 'shop'] for word in meaningful_words):
                # Try different arrangements
                brand_words = [w for w in meaningful_words if w[0].isupper()]
                other_words = [w for w in meaningful_words if w.lower() in ['wine', 'gift', 'shop', 'gifts']]
                
                if brand_words and other_words:
                    # Brand - Category - Items format
                    brand_part = ' '.join(brand_words)
                    other_part = ' '.join(other_words).title()
                    variations.append(f"{brand_part} - Shop - {other_part}")
                    variations.append(f"{brand_part} - {other_part}")
        
        return variations
    
    def _extract_brand_queries(self, query_text: str) -> List[str]:
        """Extract brand-focused queries from the original query."""
        brand_queries = []
        
        # Look for potential brand names (capitalized words)
        import re
        brand_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Multi-word capitalized terms
        ]
        
        brands = []
        for pattern in brand_patterns:
            matches = re.findall(pattern, query_text)
            brands.extend(matches)
        
        # Create brand-focused queries
        for brand in brands:
            if len(brand) > 3:
                # Brand + key terms from query
                key_terms = self._extract_key_terms(query_text).split()
                for term in key_terms:
                    if term.lower() not in brand.lower():
                        brand_query = f"{brand} {term}"
                        brand_queries.append(brand_query)
                
                # Just the brand name
                brand_queries.append(brand)
        
        return brand_queries
    
    def _create_title_variations(self, query_text: str) -> List[str]:
        """Create title-style variations that might match document titles."""
        variations = []
        
        # Extract key components
        import re
        words = re.findall(r'\b\w+\b', query_text)
        
        # Filter out question words and common terms
        meaningful_words = []
        for word in words:
            if (word.lower() not in self.question_words and 
                len(word) > 2 and 
                not word.lower() in ['about', 'the']):
                meaningful_words.append(word)
        
        if len(meaningful_words) >= 2:
            # Create title-case version
            title_case = ' '.join(word.capitalize() for word in meaningful_words)
            variations.append(title_case)
            
            # Create dash-separated version (common in titles)
            dash_version = ' - '.join(word.capitalize() for word in meaningful_words)
            variations.append(dash_version)
            
            # Create variations with common title words
            if any(word.lower() in ['wine', 'gift', 'shop'] for word in meaningful_words):
                # Try different arrangements
                brand_words = [w for w in meaningful_words if w[0].isupper()]
                other_words = [w for w in meaningful_words if w.lower() in ['wine', 'gift', 'shop', 'gifts']]
                
                if brand_words and other_words:
                    # Brand - Category - Items format
                    brand_part = ' '.join(brand_words)
                    other_part = ' '.join(other_words).title()
                    variations.append(f"{brand_part} - Shop - {other_part}")
                    variations.append(f"{brand_part} - {other_part}")
        
        return variations
    
    def _create_focused_query(self, query_text: str) -> str:
        """Create a focused query by identifying the main topic."""
        # Look for key topic indicators
        topic_patterns = [
            r'(bermuda triangle)',
            r'(\w+\s+triangle)',
            r'scientific explanation.*?(\w+)',
            r'mysterious.*?(\w+)',
            r'disappearances.*?(\w+)'
        ]
        
        for pattern in topic_patterns:
            match = re.search(pattern, query_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: take the longest meaningful phrase
        words = query_text.split()
        if len(words) > 3:
            # Find the core subject by looking for noun-like patterns
            for i in range(len(words) - 1):
                if words[i].lower() not in self.question_words and len(words[i]) > 3:
                    # Take this word and the next 1-2 words
                    end_idx = min(i + 3, len(words))
                    phrase = ' '.join(words[i:end_idx])
                    if len(phrase) > 5:
                        return phrase
        
        return query_text
    
    def _deduplicate_and_rerank(self, results: List[Dict], original_query: str, threshold: float) -> List[Dict]:
        """Remove duplicates and re-rank results based on relevance to original query."""
        if not results:
            return []
        
        # Remove duplicates based on content
        seen_content = set()
        unique_results = []
        
        for result in results:
            content_key = result.get('content', '')[:100]  # Use first 100 chars as key
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(result)
        
        # Re-rank based on multiple factors
        def calculate_relevance_score(result):
            base_score = result.get('similarity_score', 0.0)
            
            # Boost score if the result contains key terms from original query
            content = result.get('content', '').lower()
            original_lower = original_query.lower()
            
            # Check for exact phrase matches
            key_phrases = self._extract_concepts(original_query)
            phrase_boost = 0
            for phrase in key_phrases:
                if phrase.lower() in content:
                    phrase_boost += 0.1
            
            # Check for key term matches
            key_terms = self._extract_key_terms(original_query).split()
            term_boost = 0
            for term in key_terms:
                if term in content:
                    term_boost += 0.05
            
            # Boost for content length (longer content might be more informative)
            length_boost = min(0.05, len(content) / 10000)
            
            final_score = base_score + phrase_boost + term_boost + length_boost
            return min(1.0, final_score)  # Cap at 1.0
        
        # Calculate new scores and filter by threshold
        for result in unique_results:
            result['relevance_score'] = calculate_relevance_score(result)
        
        # Filter by threshold and sort by relevance score
        filtered_results = [r for r in unique_results if r['relevance_score'] >= threshold]
        filtered_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return filtered_results
    
    async def add_document(self, content: str, metadata: Dict[str, Any] = None):
        """Add a document to the vector store."""
        try:
            # Generate a document ID if not provided in metadata
            if metadata is None:
                metadata = {}
            
            if 'id' not in metadata:
                metadata['id'] = f"doc_{abs(hash(content))}"
            
            # Add timestamp
            import time
            metadata['created_at'] = time.time()
            metadata['content_length'] = len(content)
            
            embedding = self.embedding_model.encode(content).tolist()
            await vector_store.insert_embedding(
                self.default_table,
                content,
                embedding,
                metadata
            )
            logger.info("Document added to vector store", 
                       doc_id=metadata['id'], 
                       content_length=len(content))
        except Exception as e:
            logger.error("Failed to add document", error=str(e))
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the vector store."""
        try:
            # This would need to be implemented in the vector_store class
            # For now, return a mock implementation
            documents = await vector_store.list_all_documents(self.default_table)
            
            # Format documents for API response
            formatted_docs = []
            for doc in documents:
                formatted_docs.append({
                    'id': doc.get('metadata', {}).get('id', f"doc_{doc.get('id', 'unknown')}"),
                    'content': doc.get('content', '')[:200] + '...' if len(doc.get('content', '')) > 200 else doc.get('content', ''),
                    'content_length': doc.get('metadata', {}).get('content_length', len(doc.get('content', ''))),
                    'created_at': doc.get('metadata', {}).get('created_at'),
                    'metadata': doc.get('metadata', {}),
                    'source': 'vector_store'
                })
            
            return formatted_docs
            
        except Exception as e:
            logger.error("Failed to list documents", error=str(e))
            return []
    
    async def delete_document(self, doc_id: str):
        """Delete a document from the vector store."""
        try:
            # This would need to be implemented in the vector_store class
            await vector_store.delete_document(self.default_table, doc_id)
            logger.info("Document deleted from vector store", doc_id=doc_id)
        except Exception as e:
            logger.error("Failed to delete document", doc_id=doc_id, error=str(e))
            raise
    
    async def delete_all_documents(self) -> int:
        """Delete all documents from the vector store."""
        try:
            # Get count before deletion
            documents = await self.list_documents()
            count = len(documents)
            
            # Delete all documents
            await vector_store.clear_table(self.default_table)
            logger.info("All documents deleted from vector store", count=count)
            
            return count
            
        except Exception as e:
            logger.error("Failed to delete all documents", error=str(e))
            raise


vector_manager = VectorManager()
