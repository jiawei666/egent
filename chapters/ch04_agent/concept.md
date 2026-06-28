# 第 4 章：第一个 Agent（ReAct 模式）

## Agent = LLM + Tools + 循环

ReAct 循环：
1. **Reasoning**：LLM 思考下一步
2. **Acting**：调用一个 tool
3. **Observing**：把 tool 结果加入上下文
4. 重复，直到 LLM 决定任务完成

## 创建 ReAct Agent

```python
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")
tools = [get_weather, get_time]

# 从 hub 拉取标准 ReAct prompt（包含 agent_scratchpad 占位符）
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(model, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = executor.invoke({"input": "What's the weather in Beijing?"})
print(result["output"])
```

## verbose=True 的输出含义

- `Thought:` —— LLM 的推理过程
- `Action:` —— 决定调用的 tool 名
- `Action Input:` —— 传给 tool 的参数
- `Observation:` —— tool 返回的结果
- `Final Answer:` —— LLM 的最终回答

## AgentExecutor 关键参数

- `max_iterations=10` —— 防止无限循环
- `handle_parsing_errors=True` —— LLM 格式出错时重试
- `return_intermediate_steps=True` —— 返回每一步的详细信息
