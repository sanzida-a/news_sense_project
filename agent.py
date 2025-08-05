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
import requests

logfire_token = os.getenv("LOGFIRE_TOKEN")
logfire.configure(token=logfire_token)
logfire.instrument_openai_agents()

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
    topic = topic.lower()
    logfire.info("Fetching trending news", topic=topic)  # 🔧 input logged

    dummy_news = {
        "ai": [...],  # unchanged content
        "politics": [...],
        "finance": [...]
    }

    if wrapper and wrapper.context:
        preferred = wrapper.context.preferred_categories
        if topic not in preferred:
            logfire.info("Non-preferred topic", topic=topic, preferred=preferred)  # 🔧 context logged

    news_items = dummy_news.get(topic, [{"headline": f"No trending news found for {topic}", "timestamp": ""}])
    response = json.dumps({"topic": topic, "news": news_items})
    logfire.info("Trending news fetched", response=response)  # 🔧 output logged
    return response

@function_tool
async def fact_check_claim(wrapper: RunContextWrapper[UserContext], claim: str) -> str:
    claim = claim.lower()
    logfire.info("Fact checking claim", claim=claim)

    knowledge_base = {...}

    results = []
    for key in knowledge_base:
        if key in claim:
            results.extend(knowledge_base[key])

    if not results:
        results.append({"source": "System", "content": "No relevant sources found for this claim", "date": ""})

    response = json.dumps({"claim": claim, "sources": results})
    logfire.info("Fact check response", response=response)  # 🔧 output logged
    return response

@function_tool
async def summarize_news(wrapper: RunContextWrapper[UserContext], article_text: str) -> str:
    logfire.info("Summarizing article", length=len(article_text))  # 🔧 input logged

    sentences = [s.strip() for s in article_text.split('.') if s.strip()]
    summary = sentences[:3]

    key_points = []
    if len(sentences) > 0:
        key_points.append(sentences[0])
    if len(sentences) > 3:
        key_points.append(sentences[3])
    if len(sentences) > 5:
        key_points.append(sentences[5])

    response = json.dumps({
        "original_length": len(sentences),
        "summary": summary,
        "key_points": key_points if key_points else summary[:1]
    })
    logfire.info("Summarized result", response=response)  # 🔧 output logged
    return response

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
    handoffs=[
        trending_agent,        # ✅ Agent[UserContext]
        fact_checker_agent,    # ✅ Agent[UserContext]
        summarizer_agent,      # ✅ Agent[UserContext]
        conversational_agent   # ✅ Agent[UserContext]
    ]
)

async def run_demo():
    """Run demonstration of NewsSense capabilities with logging"""
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
        logfire.info("Running query", query=query)  # 🔧 Log the query

        
        result = await Runner.run(news_sense_agent, query, context=user_context)
        final_output = result.final_output
        logfire.info("Agent response", output=str(final_output))  # 🔧 Log the final output

        # Display response depending on output type
        if hasattr(final_output, "headlines"):  # Trending news
            print(f"\nTRENDING IN {final_output.topic.upper()}:")
            for i, headline in enumerate(final_output.headlines, 1):
                print(f"  {i}. {headline}")
            print(f"\nANALYSIS: {final_output.analysis}")

        elif hasattr(final_output, "verdict"):  # Fact check
            print(f"\nFACT CHECK: '{final_output.claim}'")
            print(f"VERDICT: {final_output.verdict} ({final_output.confidence} confidence)")
            print("\nSOURCES:")
            for source in final_output.sources:
                print(f"  - {source}")

        elif hasattr(final_output, "summary"):  # Summary
            print(f"\nSUMMARY (from {final_output.original_length} sentences):")
            for point in final_output.summary:
                print(f"  * {point}")
            print("\nKEY POINTS:")
            for point in final_output.key_points:
                print(f"  * {point}")

        else:  # General response
            print(f"\n{final_output}")


if __name__ == "__main__":
    asyncio.run(run_demo())