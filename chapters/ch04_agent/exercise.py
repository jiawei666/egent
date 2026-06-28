"""
第 4 章练习题

目标：构建一个能回答"北京现在几点，天气怎么样"的 Agent

要求：
  1. 使用第 3 章的 get_weather 和 get_time tools（可直接复制过来）
  2. 用 create_react_agent + AgentExecutor
  3. verbose=True，能看到 Thought/Action/Observation 循环
  4. Agent 必须实际调用两个 tool，不能直接猜测

验收标准：
  - 输出中出现 "Action: get_weather" 和 "Action: get_time"（或反过来）
  - Final Answer 包含时间和天气两个信息
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

load_dotenv()

# TODO 1: 定义 get_weather(city: str) tool（用第 3 章的版本，硬编码数据即可）

# TODO 2: 定义 get_time(timezone: str) tool（用 zoneinfo 获取真实时间）

# TODO 3: 创建 model = ChatOpenAI(...)

# TODO 4: 从 hub 拉取 prompt，创建 agent 和 executor（verbose=True）

# TODO 5: 调用 executor.invoke({"input": "北京现在几点？天气怎么样？"})
#         并打印 result["output"]

if __name__ == "__main__":
    pass  # 替换为实际代码
