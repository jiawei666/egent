"""
第 5 章练习题

目标：构建一个带 Memory 的多轮对话 Agent

场景：模拟一个旅行助手，能记住用户的偏好
  - 第一轮：用户说"我想去北京旅行，我不喜欢太热的天气"
  - 第二轮：用户问"根据我的喜好，现在适合去吗？"
  - 第三轮：用户问"我之前说不喜欢什么样的天气？"

要求：
  1. 使用 create_openai_tools_agent（支持 chat_history）
  2. 用 RunnableWithMessageHistory 包装 executor
  3. 三轮对话使用同一个 session_id

验收标准：
  - 第二轮 Agent 能结合"不喜欢热天气"和天气工具结果给出建议
  - 第三轮 Agent 能正确说出"不喜欢热的天气"
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

# TODO 1: 定义 get_weather tool（复用第 4 章版本）

# TODO 2: 创建 model、prompt（包含 chat_history MessagesPlaceholder）、agent、executor

# TODO 3: 创建 store 和 get_session_history 函数

# TODO 4: 用 RunnableWithMessageHistory 包装 executor

# TODO 5: 运行三轮对话（同一 session_id），打印每轮结果

if __name__ == "__main__":
    pass  # 替换为实际代码
