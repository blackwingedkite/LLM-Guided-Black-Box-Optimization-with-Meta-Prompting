from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.yfinance import YFinanceTools
import os
from dotenv import load_dotenv
load_dotenv()
my_api_key = os.getenv("GEMINI_API_KEY")

agent = Agent(
    model=Gemini(id="gemini-2.0-flash-001",api_key=my_api_key),
    tools=[YFinanceTools(stock_price=True)],
    instructions="Use tables to display data. Don't include any other text.",
    markdown=True,
)
agent.print_response("What is the stock price of Apple?", stream=True)