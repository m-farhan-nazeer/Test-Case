"""Router Agent - Determines optimal data sources and query strategies."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio

from agents.question_analyzer import QuestionAnalysis, QuestionType, QuestionComplexity
import structlog

logger = structlog.get_logger()


class DataSource(Enum):
    """Available data sources."""
    VECTOR_STORE = "vector_store"
    GRAPH_DB = "graph_db"
    FULL_TEXT_SEARCH = "full_text_search"
    STRUCTURED_DB = "structured_db"
    CACHE = "cache"


class QueryStrategy(Enum):
    """Query execution strategies."""
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


@dataclass
class RoutingDecision:
    """Routing decision for a question."""
    data_sources: List[DataSource]
    query_strategy: QueryStrategy
    priority_order: List[DataSource]
    timeout_seconds: int
    fallback_sources: List[DataSource]
    confidence_threshold: float


@dataclass
class QueryPlan:
    """Complete query execution plan."""
    routing_decision: RoutingDecision
    source_specific_queries: Dict[DataSource, Dict]
    expected_result_types: Dict[DataSource, str]
    synthesis_strategy: str


class RouterAgent:
    """Routes questions to optimal data sources with intelligent strategies."""
    
    def __init__(self):
        self.source_performance_history = {}
        self.load_balancing_weights = {
            DataSource.CACHE: 1.0,
            DataSource.VECTOR_STORE: 0.8,
            DataSource.FULL_TEXT_SEARCH: 0.7,
            DataSource.STRUCTURED_DB: 0.6,
            DataSource.GRAPH_DB: 0.5
        }
    
    async def route_question(self, analysis: QuestionAnalysis) -> QueryPlan:
        """Create optimal routing plan for a question."""
        logger.info("Routing question", 
                   question_type=analysis.question_type.value,
                   complexity=analysis.complexity.value)
        
        # Determine data sources
        data_sources = self._select_data_sources(analysis)
        
        # Choose query strategy
        query_strategy = self._choose_query_strategy(analysis, data_sources)
        
        # Set priorities and timeouts
        priority_order = self._prioritize_sources(data_sources, analysis)
        timeout_seconds = self._calculate_timeout(analysis.complexity)
        
        # Determine fallback sources
        fallback_sources = self._select_fallback_sources(data_sources)
        
        # Set confidence threshold
        confidence_threshold = self._determine_confidence_threshold(analysis)
        
        routing_decision = RoutingDecision(
            data_sources=data_sources,
            query_strategy=query_strategy,
            priority_order=priority_order,
            timeout_seconds=timeout_seconds,
            fallback_sources=fallback_sources,
            confidence_threshold=confidence_threshold
        )
        
        # Create source-specific queries
        source_queries = await self._create_source_queries(analysis, data_sources)
        
        # Determine expected result types
        result_types = self._determine_result_types(data_sources)
        
        # Choose synthesis strategy
        synthesis_strategy = self._choose_synthesis_strategy(analysis, data_sources)
        
        query_plan = QueryPlan(
            routing_decision=routing_decision,
            source_specific_queries=source_queries,
            expected_result_types=result_types,
            synthesis_strategy=synthesis_strategy
        )
        
        logger.info("Query plan created", 
                   sources=len(data_sources),
                   strategy=query_strategy.value)
        
        return query_plan
    
    def _select_data_sources(self, analysis: QuestionAnalysis) -> List[DataSource]:
        """Select appropriate data sources based on question analysis."""
        sources = []
        
        # Always check cache first
        sources.append(DataSource.CACHE)
        
        # Select based on search strategies
        if "vector_search" in analysis.search_strategies:
            sources.append(DataSource.VECTOR_STORE)
        
        if "full_text_search" in analysis.search_strategies:
            sources.append(DataSource.FULL_TEXT_SEARCH)
        
        if "structured_query" in analysis.search_strategies:
            sources.append(DataSource.STRUCTURED_DB)
        
        if "graph_traversal" in analysis.search_strategies:
            sources.append(DataSource.GRAPH_DB)
        
        # Question type specific logic
        if analysis.question_type == QuestionType.FACTUAL:
            if DataSource.STRUCTURED_DB not in sources:
                sources.append(DataSource.STRUCTURED_DB)
        
        elif analysis.question_type == QuestionType.ANALYTICAL:
            if DataSource.GRAPH_DB not in sources:
                sources.append(DataSource.GRAPH_DB)
            if DataSource.VECTOR_STORE not in sources:
                sources.append(DataSource.VECTOR_STORE)
        
        elif analysis.question_type == QuestionType.COMPARATIVE:
            # Comparative questions benefit from structured data
            if DataSource.STRUCTURED_DB not in sources:
                sources.append(DataSource.STRUCTURED_DB)
            if DataSource.GRAPH_DB not in sources:
                sources.append(DataSource.GRAPH_DB)
        
        # Ensure we have at least vector search as fallback
        if len(sources) == 1:  # Only cache
            sources.append(DataSource.VECTOR_STORE)
        
        return sources
    
    def _choose_query_strategy(self, analysis: QuestionAnalysis, 
                             sources: List[DataSource]) -> QueryStrategy:
        """Choose optimal query execution strategy."""
        
        # Simple questions can use parallel execution
        if analysis.complexity == QuestionComplexity.SIMPLE:
            return QueryStrategy.PARALLEL
        
        # Complex questions with many sources benefit from hierarchical
        if (analysis.complexity in [QuestionComplexity.COMPLEX, QuestionComplexity.VERY_COMPLEX] 
            and len(sources) > 3):
            return QueryStrategy.HIERARCHICAL
        
        # Multi-part questions need sequential processing
        if analysis.question_type == QuestionType.MULTI_PART:
            return QueryStrategy.SEQUENTIAL
        
        # Default to adaptive strategy
        return QueryStrategy.ADAPTIVE
    
    def _prioritize_sources(self, sources: List[DataSource], 
                          analysis: QuestionAnalysis) -> List[DataSource]:
        """Prioritize data sources based on question characteristics."""
        
        # Start with performance-based weights
        source_scores = {}
        for source in sources:
            base_score = self.load_balancing_weights.get(source, 0.5)
            
            # Adjust based on question type
            if analysis.question_type == QuestionType.FACTUAL:
                if source == DataSource.STRUCTURED_DB:
                    base_score += 0.2
                elif source == DataSource.CACHE:
                    base_score += 0.3
            
            elif analysis.question_type == QuestionType.ANALYTICAL:
                if source == DataSource.GRAPH_DB:
                    base_score += 0.3
                elif source == DataSource.VECTOR_STORE:
                    base_score += 0.2
            
            # Adjust based on entities
            if analysis.entities and source == DataSource.GRAPH_DB:
                base_score += 0.1
            
            # Adjust based on historical performance
            historical_score = self.source_performance_history.get(source, 0.5)
            final_score = (base_score * 0.7) + (historical_score * 0.3)
            
            source_scores[source] = final_score
        
        # Sort by score (highest first)
        return sorted(sources, key=lambda s: source_scores[s], reverse=True)
    
    def _calculate_timeout(self, complexity: QuestionComplexity) -> int:
        """Calculate appropriate timeout based on question complexity."""
        timeout_map = {
            QuestionComplexity.SIMPLE: 5,
            QuestionComplexity.MODERATE: 10,
            QuestionComplexity.COMPLEX: 20,
            QuestionComplexity.VERY_COMPLEX: 30
        }
        return timeout_map[complexity]
    
    def _select_fallback_sources(self, primary_sources: List[DataSource]) -> List[DataSource]:
        """Select fallback sources in case primary sources fail."""
        all_sources = list(DataSource)
        fallback = []
        
        # Add sources not in primary list
        for source in all_sources:
            if source not in primary_sources:
                fallback.append(source)
        
        # Always ensure vector store is available as ultimate fallback
        if (DataSource.VECTOR_STORE not in primary_sources and 
            DataSource.VECTOR_STORE not in fallback):
            fallback.append(DataSource.VECTOR_STORE)
        
        return fallback[:2]  # Limit to 2 fallback sources
    
    def _determine_confidence_threshold(self, analysis: QuestionAnalysis) -> float:
        """Determine minimum confidence threshold for results."""
        base_threshold = 0.7
        
        # Lower threshold for complex questions (more exploratory)
        if analysis.complexity in [QuestionComplexity.COMPLEX, QuestionComplexity.VERY_COMPLEX]:
            base_threshold -= 0.1
        
        # Higher threshold for factual questions (need accuracy)
        if analysis.question_type == QuestionType.FACTUAL:
            base_threshold += 0.1
        
        return max(0.5, min(0.9, base_threshold))
    
    async def _create_source_queries(self, analysis: QuestionAnalysis, 
                                   sources: List[DataSource]) -> Dict[DataSource, Dict]:
        """Create optimized queries for each data source."""
        queries = {}
        
        for source in sources:
            if source == DataSource.CACHE:
                queries[source] = {
                    "cache_key": self._generate_cache_key(analysis.original_question),
                    "ttl": 3600
                }
            
            elif source == DataSource.VECTOR_STORE:
                queries[source] = {
                    "query_text": analysis.original_question,
                    "limit": 10,
                    "threshold": 0.7,
                    "include_metadata": True
                }
            
            elif source == DataSource.FULL_TEXT_SEARCH:
                queries[source] = {
                    "query": " ".join(analysis.keywords),
                    "fields": ["title", "content", "summary"],
                    "size": 15,
                    "highlight": True
                }
            
            elif source == DataSource.STRUCTURED_DB:
                queries[source] = {
                    "entities": analysis.entities,
                    "keywords": analysis.keywords,
                    "question_type": analysis.question_type.value,
                    "limit": 20
                }
            
            elif source == DataSource.GRAPH_DB:
                queries[source] = {
                    "start_entities": analysis.entities,
                    "max_depth": 3,
                    "relationship_types": ["related_to", "part_of", "causes"],
                    "limit": 25
                }
        
        return queries
    
    def _determine_result_types(self, sources: List[DataSource]) -> Dict[DataSource, str]:
        """Determine expected result types from each source."""
        return {
            DataSource.CACHE: "cached_response",
            DataSource.VECTOR_STORE: "semantic_matches",
            DataSource.FULL_TEXT_SEARCH: "keyword_matches",
            DataSource.STRUCTURED_DB: "structured_data",
            DataSource.GRAPH_DB: "relationship_data"
        }
    
    def _choose_synthesis_strategy(self, analysis: QuestionAnalysis, 
                                 sources: List[DataSource]) -> str:
        """Choose strategy for synthesizing results from multiple sources."""
        
        if analysis.question_type == QuestionType.COMPARATIVE:
            return "comparative_synthesis"
        elif analysis.question_type == QuestionType.ANALYTICAL:
            return "analytical_synthesis"
        elif analysis.question_type == QuestionType.MULTI_PART:
            return "multi_part_synthesis"
        elif len(sources) > 3:
            return "weighted_synthesis"
        else:
            return "simple_synthesis"
    
    def _generate_cache_key(self, question: str) -> str:
        """Generate cache key for a question."""
        import hashlib
        return f"question:{hashlib.md5(question.encode()).hexdigest()}"
    
    async def update_source_performance(self, source: DataSource, 
                                      response_time: float, success: bool):
        """Update performance metrics for a data source."""
        if source not in self.source_performance_history:
            self.source_performance_history[source] = []
        
        # Store recent performance data (last 100 queries)
        performance_data = self.source_performance_history[source]
        performance_data.append({
            "response_time": response_time,
            "success": success,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # Keep only recent data
        if len(performance_data) > 100:
            performance_data.pop(0)
        
        # Update load balancing weights
        success_rate = sum(1 for p in performance_data if p["success"]) / len(performance_data)
        avg_response_time = sum(p["response_time"] for p in performance_data) / len(performance_data)
        
        # Calculate new weight (success rate weighted more heavily)
        new_weight = (success_rate * 0.7) + ((10 - min(avg_response_time, 10)) / 10 * 0.3)
        self.load_balancing_weights[source] = new_weight
        
        logger.info("Updated source performance", 
                   source=source.value,
                   success_rate=success_rate,
                   avg_response_time=avg_response_time,
                   new_weight=new_weight)


# Global router agent instance
router_agent = RouterAgent()
