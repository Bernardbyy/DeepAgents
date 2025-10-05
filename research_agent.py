import os
from typing import Literal

from tavily import TavilyClient

from deepagents import create_deep_agent
from prompts import (
    RESEARCH_INSTRUCTIONS as research_instructions,
    SUB_RESEARCH_PROMPT as sub_research_prompt,
    SUB_CRITIQUE_PROMPT as sub_critique_prompt,
)

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic 

load_dotenv() 

# It's best practice to initialize the client once and reuse it.
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# gemini_llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash", 
#     temperature=0,
# )

anthropic_llm = ChatAnthropic(
    model="claude-3-5-haiku-latest",
    temperature=0,
)

# Search tool to use to do research
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    search_docs = tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
    return search_docs

# Create the subagents
research_sub_agent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions. Only give this researcher one topic at a time. Do not pass multiple sub questions to this researcher. Instead, you should break down a large topic into the necessary components, and then call multiple research agents in parallel, one for each sub question.",
    "prompt": sub_research_prompt,
    "tools": [internet_search],
    "model_settings": {
        "model": "google:gemini-2.5-flash",
        "temperature": 0,
        "max_tokens": 8192
    }
}

critique_sub_agent = {
    "name": "critique-agent",
    "description": "Used to critique the final report. Give this agent some information about how you want it to critique the report.",
    "prompt": sub_critique_prompt,
    "model_settings": {
        "model": "google:gemini-2.5-flash",
        "temperature": 0,
        "max_tokens": 8192
    }
}

# Create the deep agent
agent = create_deep_agent(
    tools=[internet_search],
    instructions=research_instructions,
    subagents=[critique_sub_agent, research_sub_agent],
    model=anthropic_llm
).with_config({"recursion_limit": 1000})
