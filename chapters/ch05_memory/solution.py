"""第 5 章练习题参考答案"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city. Returns temperature and conditions."""
    data = {"Beijing": "12°C, cloudy", "Shanghai": "28°C, sunny", "Harbin": "−5°C, snowy"}
    return data.get(city, f"{city}: data not available")

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [get_weather]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Remember the user's preferences throughout the conversation."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(model, tools, prompt)
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
    config = {"configurable": {"session_id": "travel_session"}}

    r1 = agent_with_history.invoke({"input": "我想去北京旅行，我不喜欢太热的天气"}, config=config)
    print(f"第一轮: {r1['output']}\n")

    r2 = agent_with_history.invoke({"input": "根据我的喜好，北京现在适合去吗？"}, config=config)
    print(f"第二轮: {r2['output']}\n")

    r3 = agent_with_history.invoke({"input": "我之前说不喜欢什么样的天气？"}, config=config)
    print(f"第三轮: {r3['output']}")
    assert "热" in r3["output"] or "hot" in r3["output"].lower(), "Memory 失效！"
    print("✓ Memory 正常工作")
