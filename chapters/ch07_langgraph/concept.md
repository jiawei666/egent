# 第 7 章：LangGraph —— 多 Agent 协作

## 核心概念

LangGraph 把 Agent 工作流建模成有向图：
- **State**：在节点间流转的数据（用 TypedDict 定义）
- **Node**：处理 State 的函数（输入旧 State，返回 State 更新）
- **Edge**：节点之间的连接（普通边 or 条件边）

## 最简单的 StateGraph

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    message: str
    result: str

def process(state: State) -> dict:
    return {"result": state["message"].upper()}

graph = StateGraph(State)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

app = graph.compile()
result = app.invoke({"message": "hello"})
print(result["result"])  # "HELLO"
```

## 条件边（Conditional Edges）

```python
def router(state: State) -> str:
    if state["needs_more_info"]:
        return "gather_info"
    return "finalize"

graph.add_conditional_edges("decide", router, {
    "gather_info": "gather_info_node",
    "finalize": "finalize_node",
})
```

## Planner + Executor 模式

```
START → planner → executor → checker → END
                      ↑          |
                      └──────────┘ (如果未完成，继续执行)
```
