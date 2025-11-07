"""Graph Manager - Handles knowledge graph operations."""

from typing import Dict, List, Any
import asyncio

from core.database import graph_store, db_manager
from agents.router import DataSource
import structlog

logger = structlog.get_logger()


class GraphManager:
    """Manages graph database operations and traversals."""
    
    def __init__(self):
        self.db = None
    
    async def initialize(self):
        """Initialize graph collections."""
        await graph_store.create_collections()
        self.db = db_manager.arango_db
        logger.info("Graph store initialized")
    
    async def query(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute graph traversal query."""
        try:
            start_entities = query_params.get("start_entities", [])
            max_depth = query_params.get("max_depth", 3)
            relationship_types = query_params.get("relationship_types", [])
            limit = query_params.get("limit", 25)
            
            all_results = []
            
            # Traverse from each starting entity
            for entity in start_entities:
                entity_key = f"entities/{entity}"
                traversal_results = await graph_store.traverse_graph(entity_key, max_depth)
                all_results.extend(traversal_results)
            
            # Limit results
            limited_results = all_results[:limit]
            
            return {
                "source": DataSource.GRAPH_DB.value,
                "results": limited_results,
                "start_entities": start_entities,
                "total_results": len(limited_results)
            }
            
        except Exception as e:
            logger.error("Graph traversal failed", error=str(e))
            return {"source": DataSource.GRAPH_DB.value, "results": [], "error": str(e)}
    
    async def add_entity(self, entity_id: str, properties: Dict[str, Any]):
        """Add an entity to the knowledge graph."""
        try:
            entities_collection = self.db.collection('entities')
            entities_collection.insert({
                '_key': entity_id,
                **properties
            })
            logger.info("Entity added to graph", entity_id=entity_id)
        except Exception as e:
            logger.error("Failed to add entity", error=str(e))
    
    async def add_relationship(self, from_entity: str, to_entity: str, 
                             relationship_type: str, properties: Dict[str, Any] = None):
        """Add a relationship between entities."""
        try:
            relationships_collection = self.db.collection('relationships')
            relationships_collection.insert({
                '_from': f"entities/{from_entity}",
                '_to': f"entities/{to_entity}",
                'type': relationship_type,
                **(properties or {})
            })
            logger.info("Relationship added", from_entity=from_entity, to_entity=to_entity, type=relationship_type)
        except Exception as e:
            logger.error("Failed to add relationship", error=str(e))


graph_manager = GraphManager()
