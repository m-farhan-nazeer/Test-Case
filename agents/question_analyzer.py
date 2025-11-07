"""Question Analysis Agent - Parses and understands incoming questions."""

import re
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

import spacy
from langchain.schema import BaseMessage, HumanMessage
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

import structlog

logger = structlog.get_logger()


class QuestionType(Enum):
    """Types of questions the system can handle."""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    HYPOTHETICAL = "hypothetical"
    MULTI_PART = "multi_part"


class QuestionComplexity(Enum):
    """Complexity levels for questions."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class QuestionAnalysis:
    """Results of question analysis."""
    original_question: str
    question_type: QuestionType
    complexity: QuestionComplexity
    entities: List[str]
    keywords: List[str]
    intent: str
    sub_questions: List[str]
    search_strategies: List[str]
    confidence: float


class QuestionAnalyzer:
    """Analyzes questions to understand intent and extract key information."""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.llm = ChatOpenAI(temperature=1, model="gpt-5")
        self.analysis_prompt = self._create_analysis_prompt()
    
    def _create_analysis_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for question analysis."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert question analyzer. Your job is to deeply understand questions and break them down for optimal information retrieval.

Analyze the question and provide:
1. Question type (factual, analytical, comparative, procedural, causal, temporal, hypothetical, multi_part)
2. Complexity level (simple, moderate, complex, very_complex)
3. Key entities and concepts
4. Important keywords for search
5. The main intent
6. Sub-questions if the question is complex
7. Recommended search strategies
8. Confidence score (0-1)

Be thorough and precise in your analysis."""),
            ("human", "Question: {question}")
        ])
    
    async def analyze_question(self, question: str) -> QuestionAnalysis:
        """Perform comprehensive analysis of a question."""
        logger.info("Analyzing question", question=question)
        
        # Basic NLP processing
        doc = self.nlp(question)
        entities = self._extract_entities(doc)
        keywords = self._extract_keywords(doc)
        
        # LLM-based deep analysis
        llm_analysis = await self._llm_analyze(question)
        
        # Determine question type and complexity
        question_type = self._classify_question_type(question, llm_analysis)
        complexity = self._assess_complexity(question, entities, keywords)
        
        # Generate sub-questions for complex queries
        sub_questions = await self._generate_sub_questions(question, complexity)
        
        # Determine search strategies
        search_strategies = self._determine_search_strategies(question_type, entities, keywords)
        
        analysis = QuestionAnalysis(
            original_question=question,
            question_type=question_type,
            complexity=complexity,
            entities=entities,
            keywords=keywords,
            intent=llm_analysis.get("intent", ""),
            sub_questions=sub_questions,
            search_strategies=search_strategies,
            confidence=llm_analysis.get("confidence", 0.8)
        )
        
        logger.info("Question analysis complete", 
                   question_type=question_type.value,
                   complexity=complexity.value,
                   entities_count=len(entities))
        
        return analysis
    
    def _extract_entities(self, doc) -> List[str]:
        """Extract named entities from the question."""
        entities = []
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART"]:
                entities.append(ent.text)
        
        # Also extract potential brand names and compound terms
        # Look for capitalized words that might be brand names
        text = doc.text
        import re
        
        # Extract potential brand names (capitalized words, especially compound ones)
        brand_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Multi-word capitalized terms
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',       # Two-word brands
        ]
        
        for pattern in brand_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 3 and match not in entities:
                    entities.append(match)
        
        return list(set(entities))
    
    def _extract_keywords(self, doc) -> List[str]:
        """Extract important keywords from the question."""
        keywords = []
        for token in doc:
            if (token.pos_ in ["NOUN", "ADJ", "VERB"] and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 2):
                keywords.append(token.lemma_.lower())
        
        # Also extract compound terms and brand-related keywords
        text = doc.text.lower()
        
        # Extract domain-specific terms
        domain_terms = []
        if any(word in text for word in ['wine', 'winery', 'vineyard', 'cellar']):
            wine_terms = ['wine', 'winery', 'vineyard', 'cellar', 'vintage', 'tasting', 'bottle', 'gift', 'shop']
            domain_terms.extend([term for term in wine_terms if term in text])
        
        keywords.extend(domain_terms)
        
        return list(set(keywords))
    
    async def _llm_analyze(self, question: str) -> Dict:
        """Use LLM for deep question analysis."""
        try:
            # Simplified analysis without LLM for now
            question_lower = question.lower()
            
            # Determine intent based on question patterns
            if any(word in question_lower for word in ["what", "who", "where", "when"]):
                intent = "factual_query"
            elif any(word in question_lower for word in ["how", "why", "explain"]):
                intent = "analytical_query"
            elif any(word in question_lower for word in ["compare", "difference"]):
                intent = "comparative_query"
            else:
                intent = "general_query"
            
            return {
                "intent": intent,
                "confidence": 0.8
            }
        except Exception as e:
            logger.error("LLM analysis failed", error=str(e))
            return {"intent": "unknown", "confidence": 0.5}
    
    def _classify_question_type(self, question: str, llm_analysis: Dict) -> QuestionType:
        """Classify the type of question."""
        question_lower = question.lower()
        
        # Pattern matching for question types
        if any(word in question_lower for word in ["what", "who", "where", "when"]):
            return QuestionType.FACTUAL
        elif any(word in question_lower for word in ["why", "how", "explain"]):
            return QuestionType.ANALYTICAL
        elif any(word in question_lower for word in ["compare", "difference", "versus", "vs"]):
            return QuestionType.COMPARATIVE
        elif any(word in question_lower for word in ["steps", "process", "procedure"]):
            return QuestionType.PROCEDURAL
        elif any(word in question_lower for word in ["cause", "reason", "because"]):
            return QuestionType.CAUSAL
        elif any(word in question_lower for word in ["when", "before", "after", "timeline"]):
            return QuestionType.TEMPORAL
        elif any(word in question_lower for word in ["if", "suppose", "hypothetical"]):
            return QuestionType.HYPOTHETICAL
        elif len(question.split("?")) > 2 or "and" in question_lower:
            return QuestionType.MULTI_PART
        else:
            return QuestionType.FACTUAL
    
    def _assess_complexity(self, question: str, entities: List[str], keywords: List[str]) -> QuestionComplexity:
        """Assess the complexity of the question."""
        complexity_score = 0
        
        # Length factor
        if len(question.split()) > 20:
            complexity_score += 2
        elif len(question.split()) > 10:
            complexity_score += 1
        
        # Entity count factor
        complexity_score += min(len(entities), 3)
        
        # Keyword count factor
        complexity_score += min(len(keywords) // 3, 2)
        
        # Multiple questions
        if "?" in question and question.count("?") > 1:
            complexity_score += 2
        
        # Complex conjunctions
        if any(word in question.lower() for word in ["however", "moreover", "furthermore", "nevertheless"]):
            complexity_score += 1
        
        if complexity_score >= 6:
            return QuestionComplexity.VERY_COMPLEX
        elif complexity_score >= 4:
            return QuestionComplexity.COMPLEX
        elif complexity_score >= 2:
            return QuestionComplexity.MODERATE
        else:
            return QuestionComplexity.SIMPLE
    
    async def _generate_sub_questions(self, question: str, complexity: QuestionComplexity) -> List[str]:
        """Generate sub-questions for complex queries."""
        if complexity in [QuestionComplexity.SIMPLE, QuestionComplexity.MODERATE]:
            return []
        
        # Use LLM to break down complex questions
        try:
            sub_question_prompt = ChatPromptTemplate.from_messages([
                ("system", "Break down this complex question into 2-4 simpler sub-questions that together would provide a complete answer."),
                ("human", "Question: {question}")
            ])
            
            response = await self.llm.ainvoke(
                sub_question_prompt.format_messages(question=question)
            )
            
            # Parse sub-questions from response (simplified)
            sub_questions = response.content.split("\n")
            return [q.strip("- ").strip() for q in sub_questions if q.strip()][:4]
        
        except Exception as e:
            logger.error("Sub-question generation failed", error=str(e))
            return []
    
    def _determine_search_strategies(self, question_type: QuestionType, 
                                   entities: List[str], keywords: List[str]) -> List[str]:
        """Determine optimal search strategies based on question analysis."""
        strategies = []
        
        # Always include vector search for semantic similarity
        strategies.append("vector_search")
        
        # Add strategies based on question type
        if question_type == QuestionType.FACTUAL:
            strategies.extend(["full_text_search", "structured_query"])
        elif question_type == QuestionType.ANALYTICAL:
            strategies.extend(["graph_traversal", "full_text_search"])
        elif question_type == QuestionType.COMPARATIVE:
            strategies.extend(["structured_query", "graph_traversal"])
        elif question_type == QuestionType.TEMPORAL:
            strategies.extend(["structured_query", "full_text_search"])
        
        # Add graph search if entities are present
        if entities:
            strategies.append("graph_traversal")
        
        # Add full-text search if many keywords
        if len(keywords) > 3:
            strategies.append("full_text_search")
        
        return list(set(strategies))


# Global question analyzer instance
question_analyzer = QuestionAnalyzer()
