import os
import json
import asyncio
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, RunContextWrapper
import logfire

# Initialize logging
logfire.configure()
logfire.instrument_openai_agents()

# Load environment variables
load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

if not API_KEY:
    raise ValueError("API_KEY environment variable must be set")

client = AsyncOpenAI(base_url=BASE_URL if BASE_URL else None, api_key=API_KEY)

# ----------------------------- MODELS ----------------------------- #

class TrendingNewsOutput(BaseModel):
    topic: str
    headlines: List[str] = Field(description="List of trending headlines")
    analysis: str = Field(description="Brief analysis of trends")

class FactCheckOutput(BaseModel):
    claim: str
    sources: List[str] = Field(description="List of supporting sources")
    verdict: str = Field(description="Summary of fact check findings")
    confidence: str = Field(description="Confidence level in verdict")

class NewsSummaryOutput(BaseModel):
    original_length: int
    summary: List[str] = Field(description="Bullet point summary")
    key_points: List[str] = Field(description="Most important points")

@dataclass
class UserContext:
    user_id: str
    preferred_categories: List[str] = None
    session_start: datetime = None
    
    def __post_init__(self):
        if self.preferred_categories is None:
            self.preferred_categories = ["tech", "politics", "business"]
        if self.session_start is None:
            self.session_start = datetime.now()

# ----------------------------- TOOLS ----------------------------- #

@function_tool
async def get_trending_news(wrapper: RunContextWrapper[UserContext], topic: str) -> str:
    """Fetch trending news for a given topic, considering user preferences."""
    topic = topic.lower()
    logfire.info("Fetching trending news", topic=topic)
    
    # Mock data with timestamp to simulate freshness
    dummy_news = {
        "ai": [
            {"headline": "Meta releases new LLM", "timestamp": "2023-11-15T09:30:00"},
            {"headline": "Elon Musk unveils Grok update", "timestamp": "2023-11-15T08:45:00"},
            {"headline": "Google DeepMind achieves new benchmark", "timestamp": "2023-11-14T16:20:00"}
        ],
        "politics": [
            {"headline": "Election debates heat up", "timestamp": "2023-11-15T10:15:00"},
            {"headline": "New policy reforms announced", "timestamp": "2023-11-15T07:30:00"},
            {"headline": "Government tackles inflation", "timestamp": "2023-11-14T12:45:00"}
        ],
        "finance": [
            {"headline": "Stock markets rally", "timestamp": "2023-11-15T11:20:00"},
            {"headline": "Crypto prices surge", "timestamp": "2023-11-15T09:10:00"},
            {"headline": "Interest rate decisions looming", "timestamp": "2023-11-14T14:30:00"}
        ]
    }
    
    # Filter by user preferences if available
    if wrapper and wrapper.context:
        preferred = wrapper.context.preferred_categories
        if topic not in preferred:
            logfire.info("Showing non-preferred topic", topic=topic, preferred=preferred)
    
    news_items = dummy_news.get(topic, [{"headline": f"No trending news found for {topic}", "timestamp": ""}])
    return json.dumps({"topic": topic, "news": news_items})

@function_tool
async def fact_check_claim(wrapper: RunContextWrapper[UserContext], claim: str) -> str:
    """Verify claims using simulated RAG system."""
    claim = claim.lower()
    logfire.info("Fact checking claim", claim=claim)
    
    # Simulated RAG knowledge base
    knowledge_base = {
        "apple and openai": [
            {"source": "TechCrunch", "content": "Apple and OpenAI have discussed partnership", "date": "2023-11-10"},
            {"source": "Reuters", "content": "No official announcement yet regarding Apple-OpenAI", "date": "2023-11-12"}
        ],
        "meta ai investment": [
            {"source": "Bloomberg", "content": "Meta increases AI research budget by 40%", "date": "2023-11-08"},
            {"source": "The Verge", "content": "Meta denies rumors of major AI acquisition", "date": "2023-11-09"}
        ]
    }
    
    # Find matching knowledge
    results = []
    for key in knowledge_base:
        if key in claim:
            results.extend(knowledge_base[key])
    
    if not results:
        results.append({"source": "System", "content": "No relevant sources found for this claim", "date": ""})
    
    return json.dumps({"claim": claim, "sources": results})

@function_tool
async def summarize_news(wrapper: RunContextWrapper[UserContext], article_text: str) -> str:
    """Summarize news content into key points."""
    logfire.info("Summarizing article", length=len(article_text))
    
    # Simulate extractive summarization
    sentences = [s.strip() for s in article_text.split('.') if s.strip()]
    summary = sentences[:3]  # First 3 sentences as summary
    
    # Simulate key point extraction
    key_points = []
    if len(sentences) > 0:
        key_points.append(sentences[0])
    if len(sentences) > 3:
        key_points.append(sentences[3])
    if len(sentences) > 5:
        key_points.append(sentences[5])
    
    return json.dumps({
        "original_length": len(sentences),
        "summary": summary,
        "key_points": key_points if key_points else summary[:1]
    })

# ----------------------------- AGENTS ----------------------------- #

trending_agent = Agent[UserContext](
    name="Trending News Agent",
    handoff_description="Specialist agent for detecting and analyzing trending news topics",
    instructions="""
    You analyze and report on trending news topics across categories.
    
    Responsibilities:
    1. Use get_trending_news tool to fetch current headlines
    2. Group related stories and identify emerging patterns
    3. Provide brief analysis of why these topics are trending
    4. Highlight any connections between stories
    
    Output should include:
    - Clean list of headlines (with timestamps if available)
    - Brief trend analysis
    - Any notable connections between stories
    """,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[get_trending_news],
    output_type=TrendingNewsOutput
)

fact_checker_agent = Agent[UserContext](
    name="Fact Checker Agent",
    handoff_description="Specialist agent for verifying claims using RAG techniques",
    instructions="""
    You verify claims using retrieval-augmented generation (RAG).
    
    Process:
    1. Analyze the claim to identify key entities and relationships
    2. Use fact_check_claim tool to retrieve relevant sources
    3. Evaluate source reliability and recency
    4. Synthesize findings into clear verdict
    
    Output should include:
    - List of supporting/refuting sources
    - Clear verdict (Supported, Refuted, Unverified)
    - Confidence level (High, Medium, Low)
    - Brief explanation of reasoning
    """,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[fact_check_claim],
    output_type=FactCheckOutput
)

summarizer_agent = Agent[UserContext](
    name="News Summarizer Agent",
    handoff_description="Specialist agent for condensing news content into concise summaries",
    instructions="""
    You create concise, readable summaries of news content.
    
    Guidelines:
    1. Maintain key facts and context
    2. Use bullet points for readability
    3. Preserve original meaning without distortion
    4. Highlight most important 2-3 points separately
    
    Output should include:
    - Original length (word/sentence count)
    - Bullet point summary
    - Key points highlighted separately
    """,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[summarize_news],
    output_type=NewsSummaryOutput
)

conversational_agent = Agent[UserContext](
    name="Conversation Agent",
    handoff_description="Handles general conversation and routing to specialist agents",
    instructions="""
    You are the main interface for NewsSense. Your responsibilities:
    
    1. Greet users and explain capabilities
    2. Determine user intent from queries:
       - 'trending' → Trending News Agent
       - 'is/did/does' → Fact Checker Agent
       - 'summarize' → News Summarizer Agent
    3. Route to appropriate specialist agent
    4. Format final responses clearly
    
    Be friendly and informative about the system's capabilities.
    """,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client)
)

# ----------------------------- MAIN AGENT ----------------------------- #

news_sense_agent = Agent[UserContext](
    name="NewsSense Main Agent",
    instructions="""
    You are NewsSense - an AI news intelligence system that helps users:
    - Track breaking news and trends
    - Verify factual claims
    - Summarize lengthy news content
    
    You coordinate between specialized agents to provide these services.
    """,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    handoffs=[trending_agent, fact_checker_agent, summarizer_agent, conversational_agent]
)

# ----------------------------- DEMO ----------------------------- #

async def run_demo():
    """Run demonstration of NewsSense capabilities"""
    user_context = UserContext(
        user_id="news_user_123",
        preferred_categories=["tech", "politics"]
    )
    
    test_queries = [
        "What's trending in AI today?",
        "Did Apple partner with OpenAI?",
        "Summarize this article: Meta announced new AI models today. The models outperform previous versions. Experts say this advances the field significantly."
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}\nQUERY: {query}\n{'='*50}")
        
        try:
            result = await Runner.run(news_sense_agent, query, context=user_context)
            
            if hasattr(result.final_output, "headlines"):  # Trending news
                output = result.final_output
                print(f"\nTRENDING IN {output.topic.upper()}:")
                for i, headline in enumerate(output.headlines, 1):
                    print(f"  {i}. {headline}")
                print(f"\nANALYSIS: {output.analysis}")
                
            elif hasattr(result.final_output, "verdict"):  # Fact check
                output = result.final_output
                print(f"\nFACT CHECK: '{output.claim}'")
                print(f"VERDICT: {output.verdict} ({output.confidence} confidence)")
                print("\nSOURCES:")
                for source in output.sources:
                    print(f"  - {source}")
                    
            elif hasattr(result.final_output, "summary"):  # Summary
                output = result.final_output
                print(f"\nSUMMARY (from {output.original_length} sentences):")
                for point in output.summary:
                    print(f"  * {point}")
                print("\nKEY POINTS:")
                for point in output.key_points:
                    print(f"  * {point}")
                    
            else:  # General response
                print(f"\n{result.final_output}")
                
        except Exception as e:
            print(f"\nERROR: {str(e)}")
            logfire.error("Demo error", error=str(e))

if __name__ == "__main__":
    asyncio.run(run_demo())