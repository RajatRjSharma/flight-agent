import os
from crewai_tools import ApifyActorsTool
from dotenv import load_dotenv

load_dotenv()

apify_flights_tool = ApifyActorsTool(
    actor_name = "johnvc/Google-Flights-Data-Scraper-Flight-and-Price-Search",
    apify_api_token = os.getenv("APIFY_API_TOKEN")
)

TAVILY_MCP_URL = f"https://mcp.tavily.com/mcp/?tavilyApiKey={os.getenv('TAVILY_API_KEY')}"