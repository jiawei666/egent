"""第 4 章示例：第一个 ReAct Agent"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city. Returns temperature in Celsius and conditions."""
    data = {"Beijing": "12°C, cloudy", "Shanghai": "18°C, sunny", "Shenzhen": "26°C, partly cloudy"}
    return data.get(city, f"{city}: data not available")

@tool
def get_time(timezone: str) -> str:
    """Get current local time for a timezone (e.g. 'Asia/Shanghai', 'UTC')."""
    from datetime import datetime
    import zoneinfo
    try:
        now = datetime.now(zoneinfo.ZoneInfo(timezone))
        return f"Current time in {timezone}: {now.strftime('%H:%M:%S %Z')}"
    except Exception:
        return f"Unknown timezone: {timezone}"

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [get_weather, get_time]

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(model, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
)

if __name__ == "__main__":
    result = executor.invoke({"input": "What's the current time in Beijing and what's the weather like there?"})
    print("\n=== Final Answer ===")
    print(result["output"])
