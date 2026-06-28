"""
第 2 章练习题

目标：构建一个"函数文档生成器"
  输入：{"function_code": "def add(a, b): return a + b"}
  输出：DocResult 对象，包含：
    - summary: str（一句话功能说明）
    - params: list[str]（参数说明列表，格式 "name: description"）
    - returns: str（返回值说明）
    - example: str（调用示例）

要求：
  1. 定义 DocResult Pydantic 模型
  2. 用 PydanticOutputParser
  3. 用 ChatPromptTemplate.from_messages 构建 prompt
  4. chain.invoke({"function_code": "..."}) 返回的对象字段可直接访问

验收标准：result.summary / result.params / result.returns / result.example 均可访问
"""
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()
model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

# TODO 1: 定义 DocResult Pydantic 模型，包含 summary, params, returns, example 字段
class DocResult(BaseModel):
    pass  # 替换这行

# TODO 2: 创建 parser = PydanticOutputParser(pydantic_object=DocResult)
parser = None  # 替换这行

# TODO 3: 创建 prompt，system 消息要求 LLM 生成文档并包含 format_instructions
prompt = None  # 替换这行

# TODO 4: 组合 chain = prompt | model | parser
chain = None  # 替换这行

if __name__ == "__main__":
    result = chain.invoke({"function_code": "def add(a, b):\n    return a + b"})
    print(f"Summary: {result.summary}")
    print(f"Params: {result.params}")
    print(f"Returns: {result.returns}")
    print(f"Example: {result.example}")
