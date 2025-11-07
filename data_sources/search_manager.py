"""Search Manager - Handles full-text search operations."""

from typing import Dict, List, Any
import asyncio

from core.database import db_manager
from agents.router import DataSource
import structlog

logger = structlog.get_logger()


class SearchManager:
    """Manages Elasticsearch full-text search operations."""
    
    def __init__(self):
        self.es_client = None
        self.default_index = "knowledge_base"
    
    async def initialize(self):
        """Initialize Elasticsearch indices."""
        self.es_client = db_manager.elasticsearch_client
        
        # Create index if it doesn't exist
        if not await self.es_client.indices.exists(index=self.default_index):
            await self.es_client.indices.create(
                index=self.default_index,
                body={
                    "mappings": {
                        "properties": {
                            "title": {"type": "text", "analyzer": "english"},
                            "content": {"type": "text", "analyzer": "english"},
                            "summary": {"type": "text", "analyzer": "english"},
                            "keywords": {"type": "keyword"},
                            "created_at": {"type": "date"},
                            "metadata": {"type": "object"}
                        }
                    }
                }
            )
        
        logger.info("Search manager initialized")
    
    async def query(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full-text search query with enhanced matching."""
        try:
            query_text = query_params.get("query", "")
            fields = query_params.get("fields", ["title^3", "content", "summary^2"])  # Boost title matches
            size = query_params.get("size", 15)
            highlight = query_params.get("highlight", True)
            
            # Create multiple query variations for better matching
            query_variations = self._create_search_variations(query_text)
            
            # Build a more sophisticated query
            search_body = {
                "query": {
                    "bool": {
                        "should": [
                            # Exact phrase match (highest priority)
                            {
                                "multi_match": {
                                    "query": query_text,
                                    "fields": fields,
                                    "type": "phrase",
                                    "boost": 3.0
                                }
                            },
                            # Best fields match with fuzziness
                            {
                                "multi_match": {
                                    "query": query_text,
                                    "fields": fields,
                                    "type": "best_fields",
                                    "fuzziness": "AUTO",
                                    "boost": 2.0
                                }
                            },
                            # Cross fields match for related terms
                            {
                                "multi_match": {
                                    "query": query_text,
                                    "fields": fields,
                                    "type": "cross_fields",
                                    "operator": "and",
                                    "boost": 1.5
                                }
                            }
                        ]
                    }
                },
                "size": size,
                "_source": ["title", "content", "summary", "metadata", "created_at"]
            }
            
            # Add query variations as additional should clauses
            for variation in query_variations:
                if variation != query_text:
                    search_body["query"]["bool"]["should"].append({
                        "multi_match": {
                            "query": variation,
                            "fields": fields,
                            "type": "best_fields",
                            "fuzziness": "AUTO",
                            "boost": 1.0
                        }
                    })
            
            if highlight:
                search_body["highlight"] = {
                    "fields": {field.split('^')[0]: {} for field in fields}  # Remove boost notation for highlight
                }
            
            response = await self.es_client.search(
                index=self.default_index,
                body=search_body
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                result = {
                    "score": hit["_score"],
                    "source": hit["_source"],
                    "id": hit["_id"]
                }
                if highlight and "highlight" in hit:
                    result["highlights"] = hit["highlight"]
                results.append(result)
            
            return {
                "source": DataSource.FULL_TEXT_SEARCH.value,
                "results": results,
                "total_hits": response["hits"]["total"]["value"],
                "max_score": response["hits"]["max_score"],
                "query_variations_used": len(query_variations)
            }
            
        except Exception as e:
            logger.error("Full-text search failed", error=str(e))
            return {"source": DataSource.FULL_TEXT_SEARCH.value, "results": [], "error": str(e)}
    
    async def add_document(self, doc_id: str, title: str, content: str, 
                          summary: str = "", metadata: Dict[str, Any] = None):
        """Add a document to the search index."""
        try:
            doc = {
                "title": title,
                "content": content,
                "summary": summary,
                "metadata": metadata or {},
                "created_at": "now"
            }
            
            await self.es_client.index(
                index=self.default_index,
                id=doc_id,
                body=doc
            )
            
            logger.info("Document indexed", doc_id=doc_id, title=title)
        except Exception as e:
            logger.error("Failed to index document", error=str(e))
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the search index."""
        try:
            # Get all documents from the index
            response = await self.es_client.search(
                index=self.default_index,
                body={
                    "query": {"match_all": {}},
                    "size": 1000,  # Adjust as needed
                    "sort": [{"created_at": {"order": "desc"}}],
                    "_source": ["title", "content", "summary", "metadata", "created_at"]
                }
            )
            
            documents = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                content = source.get("content", "")
                
                documents.append({
                    'id': hit["_id"],
                    'title': source.get('title', ''),
                    'content': content[:200] + '...' if len(content) > 200 else content,
                    'content_length': len(content),
                    'summary': source.get('summary', ''),
                    'created_at': source.get('created_at'),
                    'metadata': source.get('metadata', {}),
                    'source': 'search_index'
                })
            
            logger.info("Listed documents from search index", count=len(documents))
            return documents
            
        except Exception as e:
            logger.error("Failed to list documents from search index", error=str(e))
            return []
    
    async def delete_document(self, doc_id: str):
        """Delete a document from the search index."""
        try:
            response = await self.es_client.delete(
                index=self.default_index,
                id=doc_id
            )
            
            if response.get("result") == "deleted":
                logger.info("Document deleted from search index", doc_id=doc_id)
            else:
                raise Exception(f"Document with ID {doc_id} not found in search index")
                
        except Exception as e:
            logger.error("Failed to delete document from search index", doc_id=doc_id, error=str(e))
            raise
    
    async def delete_all_documents(self) -> int:
        """Delete all documents from the search index."""
        try:
            # First get the count of documents
            count_response = await self.es_client.count(
                index=self.default_index,
                body={"query": {"match_all": {}}}
            )
            count = count_response["count"]
            
            # Delete all documents
            await self.es_client.delete_by_query(
                index=self.default_index,
                body={"query": {"match_all": {}}},
                refresh=True
            )
            
            logger.info("All documents deleted from search index", count=count)
            return count
            
        except Exception as e:
            logger.error("Failed to delete all documents from search index", error=str(e))
            raise
    
    def _create_search_variations(self, query_text: str) -> List[str]:
        """Create search query variations for better matching."""
        variations = [query_text]
        
        # Extract key terms
        import re
        words = re.findall(r'\b\w+\b', query_text)
        
        # Remove common question words
        question_words = {'do', 'you', 'know', 'about', 'the', 'what', 'is', 'are', 'can', 'tell', 'me'}
        meaningful_words = [word for word in words if word.lower() not in question_words and len(word) > 2]
        
        if meaningful_words:
            # Key terms only
            key_terms = ' '.join(meaningful_words)
            if key_terms != query_text:
                variations.append(key_terms)
            
            # Title case version
            title_case = ' '.join(word.capitalize() for word in meaningful_words)
            variations.append(title_case)
            
            # Brand-focused variations
            brand_words = [word for word in meaningful_words if word[0].isupper() or word.lower() in ['domaine', 'carneros']]
            if brand_words:
                brand_query = ' '.join(brand_words)
                variations.append(brand_query)
                
                # Brand + category terms
                category_words = [word for word in meaningful_words if word.lower() in ['wine', 'gift', 'gifts', 'shop']]
                if category_words:
                    brand_category = f"{' '.join(brand_words)} {' '.join(category_words)}"
                    variations.append(brand_category)
        
        return list(set(variations))  # Remove duplicates


search_manager = SearchManager()
