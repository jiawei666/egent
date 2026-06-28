"""第 5 章示例：给 Agent 加上多轮对话记忆"""
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    data = {"Beijing": "12°C, cloudy", "Shanghai": "18°C, sunny"}
    return data.get(city, f"{city}: data not available")

model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
tools = [get_weather]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(model, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

agent_with_history = RunnableWithMessageHistory(
    executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

if __name__ == "__main__":
    config = {"configurable": {"session_id": "demo_session"}}

    r1 = agent_with_history.invoke({"input": "帮我查一下北京的天气"}, config=config)
    print(f"第一轮: {r1['output']}")

    r2 = agent_with_history.invoke({"input": "刚才你查的是哪个城市？"}, config=config)
    print(f"第二轮: {r2['output']}")
    assert "beijing" in r2["output"].lower() or "北京" in r2["output"], \
        "Agent 没能记住上一轮的城市！"
    print("✓ Memory 正常工作")
