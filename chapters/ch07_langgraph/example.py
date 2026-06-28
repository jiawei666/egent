"""第 7 章示例：Planner + Executor 双 Agent 系统"""
from dotenv import load_dotenv
from typing import TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

load_dotenv()

model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
parser = StrOutputParser()

class State(TypedDict):
    task: str
    plan: list[str]
    current_step: int
    results: list[str]
    final_answer: str

@tool
def search_web(query: str) -> str:
    """Search the web for information about a topic."""
    results = {
        "Python web frameworks": "Top frameworks: FastAPI (modern, fast), Django (batteries-included), Flask (lightweight)",
        "FastAPI features": "FastAPI: async support, auto OpenAPI docs, Pydantic validation, high performance",
        "Django features": "Django: ORM, admin panel, authentication, mature ecosystem, 'batteries included'",
        "Flask features": "Flask: minimal, flexible, easy to learn, large extension ecosystem",
    }
    for key, value in results.items():
        if any(word.lower() in query.lower() for word in key.split()):
            return value
    return f"Search results for '{query}': Found general information about the topic."

planner_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a task planner. Break down the given task into 3-4 concrete steps. Return ONLY a numbered list, one step per line."),
        ("human", "Task: {task}"),
    ])
    | model
    | parser
)

executor_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a task executor. Execute the given step and provide a concise result. Use the search tool if needed."),
        ("human", "Task: {task}\nStep to execute: {step}\nPrevious results: {previous_results}"),
    ])
    | model.bind_tools([search_web])
    | parser
)

def planner_node(state: State) -> dict:
    print(f"\n[Planner] Breaking down task: {state['task']}")
    plan_text = planner_chain.invoke({"task": state["task"]})
    steps = [line.strip() for line in plan_text.split("\n") if line.strip() and line.strip()[0].isdigit()]
    steps = [s.split(". ", 1)[1] if ". " in s else s for s in steps]
    print(f"[Planner] Plan: {steps}")
    return {"plan": steps, "current_step": 0, "results": []}

def executor_node(state: State) -> dict:
    step = state["plan"][state["current_step"]]
    print(f"\n[Executor] Executing step {state['current_step'] + 1}: {step}")
    result = executor_chain.invoke({
        "task": state["task"],
        "step": step,
        "previous_results": "\n".join(state["results"]) if state["results"] else "None",
    })
    print(f"[Executor] Result: {result[:100]}...")
    return {
        "results": state["results"] + [f"Step {state['current_step'] + 1}: {result}"],
        "current_step": state["current_step"] + 1,
    }

def finalizer_node(state: State) -> dict:
    summary_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "Summarize the research results into a clear, structured answer."),
            ("human", "Task: {task}\nResults:\n{results}"),
        ])
        | model
        | parser
    )
    final = summary_chain.invoke({"task": state["task"], "results": "\n".join(state["results"])})
    return {"final_answer": final}

def should_continue(state: State) -> str:
    if state["current_step"] < len(state["plan"]):
        return "execute"
    return "finalize"

graph = StateGraph(State)
graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("finalizer", finalizer_node)
graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", should_continue, {"execute": "executor", "finalize": "finalizer"})
graph.add_edge("finalizer", END)

app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({"task": "Research the top 3 Python web frameworks and their key features", "plan": [], "current_step": 0, "results": [], "final_answer": ""})
    print("\n=== Final Answer ===")
    print(result["final_answer"])
