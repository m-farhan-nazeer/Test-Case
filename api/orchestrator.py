"""Query Orchestrator - Coordinates the entire question processing pipeline."""

import asyncio
from typing import Dict, Any
import time

from agents.question_analyzer import question_analyzer
from agents.router import router_agent, DataSource
from data_sources.vector_manager import vector_manager
from data_sources.graph_manager import graph_manager
from data_sources.search_manager import search_manager
from data_sources.cache_manager import cache_manager
from synthesis.synthesizer import knowledge_synthesizer, SynthesisResult
from config.settings import settings
import structlog

logger = structlog.get_logger()


class QueryOrchestrator:
    """Orchestrates the complete question processing pipeline."""
    
    def __init__(self):
        self.source_managers = {
            DataSource.VECTOR_STORE: vector_manager,
            DataSource.GRAPH_DB: graph_manager,
            DataSource.FULL_TEXT_SEARCH: search_manager,
            DataSource.CACHE: cache_manager
        }
    
    async def process_question(self, question: str, context: Dict[str, Any], config: Dict[str, Any] = None) -> SynthesisResult:
        """Process a question through the complete pipeline."""
        logger.info("Starting question processing pipeline", question=question)
        
        # Use provided config or defaults
        if config is None:
            config = {
                "confidence_threshold": settings.confidence_threshold,
                "max_results": settings.max_results,
                "timeout": settings.query_timeout
            }
        
        try:
            # Step 1: Analyze the question
            analysis = await question_analyzer.analyze_question(question)
            logger.info("Question analysis complete", 
                       question_type=analysis.question_type.value,
                       complexity=analysis.complexity.value)
            
            # Step 2: Create routing plan with config
            query_plan = await router_agent.route_question(analysis, config)
            logger.info("Query plan created", 
                       sources=len(query_plan.routing_decision.data_sources),
                       strategy=query_plan.routing_decision.query_strategy.value)
            
            # Step 3: Execute queries based on strategy
            source_results = await self._execute_queries(query_plan, config)
            
            # Step 4: Synthesize results
            synthesis_result = await knowledge_synthesizer.synthesize(
                analysis, 
                source_results, 
                query_plan.synthesis_strategy
            )
            
            # Step 5: Cache the result if it's good quality
            cache_threshold = config.get("confidence_threshold", settings.confidence_threshold)
            if synthesis_result.confidence > cache_threshold:
                cache_key = f"question:{hash(question)}"
                await cache_manager.set_cache(cache_key, {
                    "answer": synthesis_result.answer,
                    "confidence": synthesis_result.confidence,
                    "citations": synthesis_result.citations,
                    "timestamp": time.time()
                })
            
            logger.info("Question processing complete", 
                       confidence=synthesis_result.confidence,
                       sources_used=len(synthesis_result.sources_used))
            
            return synthesis_result
            
        except Exception as e:
            logger.error("Question processing failed", error=str(e))
            return SynthesisResult(
                answer="I encountered an error while processing your question. Please try again.",
                confidence=0.0,
                sources_used=[],
                source_contributions={},
                citations=[],
                reasoning=f"Processing error: {str(e)}"
            )
    
    async def _execute_queries(self, query_plan, config: Dict[str, Any]) -> Dict[DataSource, Dict[str, Any]]:
        """Execute queries against data sources based on the query plan."""
        routing_decision = query_plan.routing_decision
        source_queries = query_plan.source_specific_queries
        
        # Inject config into source queries
        for source, query_params in source_queries.items():
            query_params.update({
                "threshold": config.get("confidence_threshold", settings.confidence_threshold),
                "limit": config.get("max_results", settings.max_results),
                "timeout": config.get("timeout", settings.query_timeout)
            })
        
        if routing_decision.query_strategy.value == "parallel":
            return await self._execute_parallel_queries(routing_decision, source_queries)
        elif routing_decision.query_strategy.value == "sequential":
            return await self._execute_sequential_queries(routing_decision, source_queries)
        elif routing_decision.query_strategy.value == "hierarchical":
            return await self._execute_hierarchical_queries(routing_decision, source_queries)
        else:  # adaptive
            return await self._execute_adaptive_queries(routing_decision, source_queries)
    
    async def _execute_parallel_queries(self, routing_decision, source_queries) -> Dict[DataSource, Dict[str, Any]]:
        """Execute all queries in parallel."""
        tasks = []
        sources = []
        
        for source in routing_decision.priority_order:
            if source in source_queries and source in self.source_managers:
                manager = self.source_managers[source]
                query_params = source_queries[source]
                
                task = asyncio.create_task(
                    self._execute_single_query(manager, source, query_params, routing_decision.timeout_seconds)
                )
                tasks.append(task)
                sources.append(source)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        source_results = {}
        for source, result in zip(sources, results):
            if isinstance(result, Exception):
                logger.error("Query failed", source=source.value, error=str(result))
                source_results[source] = {"results": [], "error": str(result)}
            else:
                source_results[source] = result
        
        return source_results
    
    async def _execute_sequential_queries(self, routing_decision, source_queries) -> Dict[DataSource, Dict[str, Any]]:
        """Execute queries sequentially, using results from previous queries."""
        source_results = {}
        
        for source in routing_decision.priority_order:
            if source in source_queries and source in self.source_managers:
                manager = self.source_managers[source]
                query_params = source_queries[source]
                
                try:
                    result = await self._execute_single_query(
                        manager, source, query_params, routing_decision.timeout_seconds
                    )
                    source_results[source] = result
                    
                    # Early termination if we get high-confidence results
                    if (source == DataSource.CACHE and 
                        result.get("cache_hit") and 
                        len(result.get("results", [])) > 0):
                        logger.info("Early termination due to cache hit")
                        break
                        
                except Exception as e:
                    logger.error("Sequential query failed", source=source.value, error=str(e))
                    source_results[source] = {"results": [], "error": str(e)}
        
        return source_results
    
    async def _execute_hierarchical_queries(self, routing_decision, source_queries) -> Dict[DataSource, Dict[str, Any]]:
        """Execute queries in hierarchical order - fast sources first, then detailed sources."""
        source_results = {}
        
        # First tier: Fast sources (cache, structured DB)
        fast_sources = [DataSource.CACHE, DataSource.STRUCTURED_DB]
        fast_tasks = []
        fast_source_list = []
        
        for source in fast_sources:
            if source in routing_decision.data_sources and source in source_queries:
                manager = self.source_managers[source]
                query_params = source_queries[source]
                
                task = asyncio.create_task(
                    self._execute_single_query(manager, source, query_params, 5)  # Short timeout
                )
                fast_tasks.append(task)
                fast_source_list.append(source)
        
        if fast_tasks:
            fast_results = await asyncio.gather(*fast_tasks, return_exceptions=True)
            for source, result in zip(fast_source_list, fast_results):
                if not isinstance(result, Exception):
                    source_results[source] = result
        
        # Check if we have sufficient results from fast sources
        total_fast_results = sum(len(result.get("results", [])) for result in source_results.values())
        
        if total_fast_results < 3:  # Need more results
            # Second tier: Comprehensive sources
            comprehensive_sources = [DataSource.VECTOR_STORE, DataSource.FULL_TEXT_SEARCH, DataSource.GRAPH_DB]
            comp_tasks = []
            comp_source_list = []
            
            for source in comprehensive_sources:
                if source in routing_decision.data_sources and source in source_queries:
                    manager = self.source_managers[source]
                    query_params = source_queries[source]
                    
                    task = asyncio.create_task(
                        self._execute_single_query(manager, source, query_params, routing_decision.timeout_seconds)
                    )
                    comp_tasks.append(task)
                    comp_source_list.append(source)
            
            if comp_tasks:
                comp_results = await asyncio.gather(*comp_tasks, return_exceptions=True)
                for source, result in zip(comp_source_list, comp_results):
                    if not isinstance(result, Exception):
                        source_results[source] = result
        
        return source_results
    
    async def _execute_adaptive_queries(self, routing_decision, source_queries) -> Dict[DataSource, Dict[str, Any]]:
        """Execute queries adaptively based on intermediate results."""
        source_results = {}
        
        # Start with highest priority source
        for source in routing_decision.priority_order[:2]:  # Top 2 sources first
            if source in source_queries and source in self.source_managers:
                manager = self.source_managers[source]
                query_params = source_queries[source]
                
                try:
                    result = await self._execute_single_query(
                        manager, source, query_params, routing_decision.timeout_seconds // 2
                    )
                    source_results[source] = result
                    
                    # Evaluate if we need more sources
                    result_count = len(result.get("results", []))
                    if result_count >= 5:  # Sufficient results
                        logger.info("Adaptive strategy: sufficient results from primary sources")
                        break
                        
                except Exception as e:
                    logger.error("Adaptive query failed", source=source.value, error=str(e))
                    source_results[source] = {"results": [], "error": str(e)}
        
        # If we don't have enough results, query remaining sources
        total_results = sum(len(result.get("results", [])) for result in source_results.values())
        if total_results < 3:
            remaining_sources = [s for s in routing_decision.priority_order[2:] if s not in source_results]
            
            for source in remaining_sources:
                if source in source_queries and source in self.source_managers:
                    manager = self.source_managers[source]
                    query_params = source_queries[source]
                    
                    try:
                        result = await self._execute_single_query(
                            manager, source, query_params, routing_decision.timeout_seconds
                        )
                        source_results[source] = result
                    except Exception as e:
                        logger.error("Adaptive fallback query failed", source=source.value, error=str(e))
                        source_results[source] = {"results": [], "error": str(e)}
        
        return source_results
    
    async def _execute_single_query(self, manager, source: DataSource, 
                                   query_params: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        """Execute a single query with timeout and performance tracking."""
        start_time = time.time()
        
        try:
            # Execute query with timeout
            result = await asyncio.wait_for(
                manager.query(query_params),
                timeout=timeout
            )
            
            # Track performance
            response_time = time.time() - start_time
            await router_agent.update_source_performance(source, response_time, True)
            
            logger.info("Query executed successfully", 
                       source=source.value,
                       response_time=response_time,
                       result_count=len(result.get("results", [])))
            
            return result
            
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            await router_agent.update_source_performance(source, response_time, False)
            logger.warning("Query timeout", source=source.value, timeout=timeout)
            raise
            
        except Exception as e:
            response_time = time.time() - start_time
            await router_agent.update_source_performance(source, response_time, False)
            logger.error("Query execution failed", source=source.value, error=str(e))
            raise


# Global orchestrator instance
query_orchestrator = QueryOrchestrator()
