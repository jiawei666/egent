"""第 8 章练习题参考答案"""
import asyncio
from dotenv import load_dotenv
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

load_dotenv()

# 如需开启 LangSmith tracing，在 .env 中设置：
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your_api_key
# LANGCHAIN_PROJECT=langchain-learning

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

class State(TypedDict):
    task: str
    plan: list[str]
    current_step: int
    results: list[str]
    final_answer: str

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    data = {
        "Python web frameworks": "Top: FastAPI, Django, Flask",
        "FastAPI": "Fast, async, auto-docs, Pydantic",
        "Django": "Batteries-included, ORM, admin",
        "Flask": "Minimal, flexible, easy to learn",
    }
    for k, v in data.items():
        if k.lower() in query.lower():
            return v
    return f"General info about: {query}"

planner_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Break down the task into 3 concrete steps. Return ONLY a numbered list."),
        ("human", "Task: {task}"),
    ]) | model | parser
)

def planner_node(state: State) -> dict:
    print(f"\n[Planner] Planning: {state['task']}")
    text = planner_chain.invoke({"task": state["task"]})
    steps = [l.split(". ", 1)[1] if ". " in l else l for l in text.split("\n") if l.strip() and l.strip()[0].isdigit()]
    return {"plan": steps, "current_step": 0, "results": []}

def executor_node(state: State) -> dict:
    step = state["plan"][state["current_step"]]
    print(f"\n[Executor] Step {state['current_step'] + 1}: {step}")
    try:
        result = search_web.invoke({"query": step})
    except Exception as e:
        result = f"工具调用失败（{type(e).__name__}），跳过此步骤"
    return {
        "results": state["results"] + [f"Step {state['current_step'] + 1}: {result}"],
        "current_step": state["current_step"] + 1,
    }

def finalizer_node(state: State) -> dict:
    summary_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "Summarize the research into a clear answer."),
            ("human", "Task: {task}\nResults:\n{results}"),
        ]) | model | parser
    )
    final = summary_chain.invoke({"task": state["task"], "results": "\n".join(state["results"])})
    return {"final_answer": final}

def should_continue(state: State) -> str:
    return "execute" if state["current_step"] < len(state["plan"]) else "finalize"

graph = StateGraph(State)
graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("finalizer", finalizer_node)
graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", should_continue, {"execute": "executor", "finalize": "finalizer"})
graph.add_edge("finalizer", END)
app = graph.compile()

async def main():
    print("=== 流式输出（每个节点完成时显示）===")
    async for chunk in app.astream(
        {"task": "Research Python web frameworks", "plan": [], "current_step": 0, "results": [], "final_answer": ""}
    ):
        node_name = list(chunk.keys())[0]
        print(f"[节点 {node_name} 完成]")

    print("\n=== 异步完整调用 ===")
    result = await app.ainvoke(
        {"task": "Research Python web frameworks", "plan": [], "current_step": 0, "results": [], "final_answer": ""}
    )
    print("\n=== Final Answer ===")
    print(result["final_answer"])

if __name__ == "__main__":
    asyncio.run(main())
