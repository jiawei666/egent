"""第 8 章示例：流式输出 + 异步 + 优雅降级"""
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
    ])
    | model
    | parser
)

def demo_streaming():
    """流式输出示例"""
    print("=== 流式输出 ===")
    for chunk in chain.stream({"input": "用 3 句话解释什么是机器学习"}):
        print(chunk, end="", flush=True)
    print("\n")

async def demo_async():
    """异步调用示例"""
    print("=== 异步调用 ===")
    result = await chain.ainvoke({"input": "Python 有哪些优点？用一句话"})
    print(result)
    print()

def demo_fallback():
    """优雅降级示例"""
    print("=== 优雅降级 ===")

    def always_fail(x):
        raise ConnectionError("模拟网络超时")

    def handle_error(error: Exception) -> str:
        return f"工具调用失败（{type(error).__name__}），已使用默认回答：暂无相关数据。"

    safe_chain = RunnableLambda(always_fail).with_fallbacks(
        [RunnableLambda(lambda x: handle_error(ConnectionError("模拟错误")))]
    )
    result = safe_chain.invoke({"query": "test"})
    print(result)
    print()

if __name__ == "__main__":
    demo_streaming()
    asyncio.run(demo_async())
    demo_fallback()
    print("✓ 所有生产化功能正常工作")
    print("\n提示：在 .env 中设置 LANGCHAIN_TRACING_V2=true 和 LANGCHAIN_API_KEY")
    print("即可在 https://smith.langchain.com 查看完整 trace")
