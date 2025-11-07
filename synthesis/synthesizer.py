"""Knowledge Synthesizer - Combines results from multiple data sources."""

from typing import Dict, List, Any, Optional
import asyncio
from dataclasses import dataclass

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from agents.question_analyzer import QuestionAnalysis, QuestionType
from agents.router import DataSource
import structlog

logger = structlog.get_logger()


@dataclass
class SynthesisResult:
    """Result of knowledge synthesis."""
    answer: str
    confidence: float
    sources_used: List[DataSource]
    source_contributions: Dict[DataSource, float]
    citations: List[Dict[str, Any]]
    reasoning: str


class KnowledgeSynthesizer:
    """Synthesizes knowledge from multiple data sources into coherent answers."""
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=1, model="gpt-5")
        self.synthesis_prompts = self._create_synthesis_prompts()
    
    def _create_synthesis_prompts(self) -> Dict[str, ChatPromptTemplate]:
        """Create synthesis prompt templates for different strategies."""
        prompts = {}
        
        # Simple synthesis
        prompts["simple_synthesis"] = ChatPromptTemplate.from_messages([
            ("system", """You are an expert knowledge synthesizer. Combine information from multiple sources to provide a comprehensive, accurate answer.

Guidelines:
- Synthesize information coherently
- Resolve any conflicts by weighing source reliability
- Include relevant citations
- Provide confidence assessment
- Be concise but complete"""),
            ("human", "Question: {question}\n\nSources:\n{sources}\n\nProvide a synthesized answer.")
        ])
        
        # Comparative synthesis
        prompts["comparative_synthesis"] = ChatPromptTemplate.from_messages([
            ("system", """You are synthesizing information for a comparative question. Focus on:
- Clear comparisons and contrasts
- Balanced presentation of different perspectives
- Structured analysis of similarities and differences
- Evidence-based conclusions"""),
            ("human", "Question: {question}\n\nSources:\n{sources}\n\nProvide a comparative analysis.")
        ])
        
        # Analytical synthesis
        prompts["analytical_synthesis"] = ChatPromptTemplate.from_messages([
            ("system", """You are synthesizing information for an analytical question. Focus on:
- Deep analysis and interpretation
- Cause-and-effect relationships
- Multiple perspectives and nuances
- Evidence-based reasoning"""),
            ("human", "Question: {question}\n\nSources:\n{sources}\n\nProvide an analytical synthesis.")
        ])
        
        return prompts
    
    async def synthesize(self, question_analysis: QuestionAnalysis, 
                        source_results: Dict[DataSource, Dict[str, Any]], 
                        synthesis_strategy: str) -> SynthesisResult:
        """Synthesize results from multiple data sources."""
        logger.info("Starting knowledge synthesis", 
                   strategy=synthesis_strategy,
                   sources=list(source_results.keys()))
        
        # Filter out empty results and errors
        valid_results = {
            source: result for source, result in source_results.items()
            if result.get("results") and not result.get("error")
        }
        
        if not valid_results:
            return SynthesisResult(
                answer="I couldn't find sufficient information to answer your question.",
                confidence=0.0,
                sources_used=[],
                source_contributions={},
                citations=[],
                reasoning="No valid results from any data source."
            )
        
        # Check for cached result first
        if DataSource.CACHE in valid_results and valid_results[DataSource.CACHE].get("cache_hit"):
            cached_data = valid_results[DataSource.CACHE]["results"][0]
            return SynthesisResult(
                answer=cached_data.get("answer", ""),
                confidence=cached_data.get("confidence", 0.8),
                sources_used=[DataSource.CACHE],
                source_contributions={DataSource.CACHE: 1.0},
                citations=cached_data.get("citations", []),
                reasoning="Retrieved from cache"
            )
        
        # Prepare sources for synthesis
        formatted_sources = self._format_sources_for_synthesis(valid_results)
        
        # Choose synthesis approach
        if synthesis_strategy == "weighted_synthesis":
            return await self._weighted_synthesis(question_analysis, valid_results, formatted_sources)
        elif synthesis_strategy == "multi_part_synthesis":
            return await self._multi_part_synthesis(question_analysis, valid_results, formatted_sources)
        else:
            return await self._standard_synthesis(question_analysis, valid_results, formatted_sources, synthesis_strategy)
    
    def _format_sources_for_synthesis(self, source_results: Dict[DataSource, Dict[str, Any]]) -> str:
        """Format source results for LLM synthesis."""
        formatted = []
        
        for source, result in source_results.items():
            if source == DataSource.CACHE:
                continue
                
            source_info = f"\n--- {source.value.upper()} ---\n"
            
            if source == DataSource.VECTOR_STORE:
                for i, item in enumerate(result["results"][:5]):  # Top 5 results
                    source_info += f"{i+1}. {item['content']} (similarity: {item['similarity']:.2f})\n"
            
            elif source == DataSource.FULL_TEXT_SEARCH:
                for i, item in enumerate(result["results"][:5]):
                    source_info += f"{i+1}. {item['source']['title']}: {item['source']['content'][:200]}...\n"
            
            elif source == DataSource.GRAPH_DB:
                for i, item in enumerate(result["results"][:5]):
                    vertex = item.get("vertex", {})
                    source_info += f"{i+1}. Entity: {vertex.get('name', 'Unknown')} - {vertex.get('description', '')}\n"
            
            elif source == DataSource.STRUCTURED_DB:
                for i, item in enumerate(result["results"][:5]):
                    source_info += f"{i+1}. {str(item)[:200]}...\n"
            
            formatted.append(source_info)
        
        return "\n".join(formatted)
    
    async def _standard_synthesis(self, question_analysis: QuestionAnalysis, 
                                valid_results: Dict[DataSource, Dict[str, Any]], 
                                formatted_sources: str, synthesis_strategy: str) -> SynthesisResult:
        """Perform standard synthesis without LLM dependency."""
        try:
            # Simple rule-based synthesis for now
            question = question_analysis.original_question
            
            # Collect all content from sources
            all_content = []
            for source, result in valid_results.items():
                if source == DataSource.CACHE:
                    continue
                for item in result.get("results", []):
                    if isinstance(item, dict) and "content" in item:
                        all_content.append(item["content"])
            
            if not all_content:
                return SynthesisResult(
                    answer="I don't have any documents in my knowledge base yet. Please upload some documents first using the 'Upload Documents' tab, then I'll be able to provide informed answers based on your content.",
                    confidence=0.1,
                    sources_used=[],
                    source_contributions={},
                    citations=[],
                    reasoning="No documents available in knowledge base"
                )
            
            # Create a simple answer by combining the most relevant content
            answer = f"Based on the available documents, here's what I found regarding '{question}':\n\n"
            
            # Add top 3 pieces of content
            for i, content in enumerate(all_content[:3]):
                answer += f"{i+1}. {content[:200]}{'...' if len(content) > 200 else ''}\n\n"
            
            # Calculate source contributions
            source_contributions = self._calculate_source_contributions(valid_results)
            
            # Extract citations
            citations = self._extract_citations(valid_results)
            
            # Calculate confidence
            confidence = self._calculate_confidence(valid_results, source_contributions)
            
            return SynthesisResult(
                answer=answer,
                confidence=confidence,
                sources_used=list(valid_results.keys()),
                source_contributions=source_contributions,
                citations=citations,
                reasoning=f"Synthesized using {synthesis_strategy} from {len(valid_results)} sources with {len(all_content)} documents"
            )
            
        except Exception as e:
            logger.error("Synthesis failed", error=str(e))
            return SynthesisResult(
                answer="I encountered an error while processing your question.",
                confidence=0.0,
                sources_used=[],
                source_contributions={},
                citations=[],
                reasoning=f"Synthesis error: {str(e)}"
            )
    
    async def _weighted_synthesis(self, question_analysis: QuestionAnalysis, 
                                valid_results: Dict[DataSource, Dict[str, Any]], 
                                formatted_sources: str) -> SynthesisResult:
        """Perform weighted synthesis based on source reliability."""
        # Weight sources based on reliability and relevance
        source_weights = {
            DataSource.VECTOR_STORE: 0.9,  # High semantic relevance
            DataSource.STRUCTURED_DB: 0.8,  # High factual accuracy
            DataSource.FULL_TEXT_SEARCH: 0.7,  # Good keyword matching
            DataSource.GRAPH_DB: 0.6,  # Good for relationships
        }
        
        # Adjust weights based on question type
        if question_analysis.question_type == QuestionType.FACTUAL:
            source_weights[DataSource.STRUCTURED_DB] = 0.95
        elif question_analysis.question_type == QuestionType.ANALYTICAL:
            source_weights[DataSource.GRAPH_DB] = 0.85
            source_weights[DataSource.VECTOR_STORE] = 0.9
        
        # Use standard synthesis with weighted consideration
        return await self._standard_synthesis(question_analysis, valid_results, formatted_sources, "simple_synthesis")
    
    async def _multi_part_synthesis(self, question_analysis: QuestionAnalysis, 
                                  valid_results: Dict[DataSource, Dict[str, Any]], 
                                  formatted_sources: str) -> SynthesisResult:
        """Handle multi-part questions by addressing each part."""
        if not question_analysis.sub_questions:
            return await self._standard_synthesis(question_analysis, valid_results, formatted_sources, "simple_synthesis")
        
        # Address each sub-question
        sub_answers = []
        for i, sub_question in enumerate(question_analysis.sub_questions):
            sub_prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer this specific sub-question using the provided sources."),
                ("human", f"Sub-question {i+1}: {sub_question}\n\nSources:\n{formatted_sources}")
            ])
            
            try:
                response = await self.llm.ainvoke(sub_prompt.format_messages())
                sub_answers.append(f"{i+1}. {sub_question}\n{response.content}\n")
            except Exception as e:
                logger.error("Sub-question synthesis failed", sub_question=sub_question, error=str(e))
                sub_answers.append(f"{i+1}. {sub_question}\nUnable to answer this part.\n")
        
        # Combine sub-answers
        final_answer = "Here's a comprehensive answer addressing each part of your question:\n\n" + "\n".join(sub_answers)
        
        source_contributions = self._calculate_source_contributions(valid_results)
        citations = self._extract_citations(valid_results)
        confidence = self._calculate_confidence(valid_results, source_contributions)
        
        return SynthesisResult(
            answer=final_answer,
            confidence=confidence,
            sources_used=list(valid_results.keys()),
            source_contributions=source_contributions,
            citations=citations,
            reasoning=f"Multi-part synthesis addressing {len(question_analysis.sub_questions)} sub-questions"
        )
    
    def _calculate_source_contributions(self, valid_results: Dict[DataSource, Dict[str, Any]]) -> Dict[DataSource, float]:
        """Calculate how much each source contributed to the answer."""
        total_results = sum(len(result.get("results", [])) for result in valid_results.values())
        
        if total_results == 0:
            return {}
        
        contributions = {}
        for source, result in valid_results.items():
            result_count = len(result.get("results", []))
            contributions[source] = result_count / total_results
        
        return contributions
    
    def _extract_citations(self, valid_results: Dict[DataSource, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract citation information from source results."""
        citations = []
        
        for source, result in valid_results.items():
            if source == DataSource.CACHE:
                continue
                
            for item in result.get("results", [])[:3]:  # Top 3 per source
                citation = {
                    "source": source.value,
                    "content": str(item)[:100] + "..." if len(str(item)) > 100 else str(item)
                }
                
                if source == DataSource.VECTOR_STORE:
                    citation["similarity"] = item.get("similarity", 0)
                elif source == DataSource.FULL_TEXT_SEARCH:
                    citation["score"] = item.get("score", 0)
                    citation["title"] = item.get("source", {}).get("title", "")
                
                citations.append(citation)
        
        return citations
    
    def _calculate_confidence(self, valid_results: Dict[DataSource, Dict[str, Any]], 
                            source_contributions: Dict[DataSource, float]) -> float:
        """Calculate overall confidence in the synthesized answer."""
        if not valid_results:
            return 0.0
        
        # Base confidence on number of sources
        source_count_factor = min(len(valid_results) / 3, 1.0)  # Max benefit from 3 sources
        
        # Factor in source quality
        quality_scores = {
            DataSource.VECTOR_STORE: 0.9,
            DataSource.STRUCTURED_DB: 0.95,
            DataSource.FULL_TEXT_SEARCH: 0.8,
            DataSource.GRAPH_DB: 0.85,
            DataSource.CACHE: 0.9
        }
        
        weighted_quality = sum(
            quality_scores.get(source, 0.5) * contribution
            for source, contribution in source_contributions.items()
        )
        
        # Factor in result counts
        total_results = sum(len(result.get("results", [])) for result in valid_results.values())
        result_count_factor = min(total_results / 10, 1.0)  # Max benefit from 10 results
        
        # Combine factors
        confidence = (source_count_factor * 0.4 + weighted_quality * 0.4 + result_count_factor * 0.2)
        
        return min(max(confidence, 0.1), 0.95)  # Clamp between 0.1 and 0.95


knowledge_synthesizer = KnowledgeSynthesizer()
