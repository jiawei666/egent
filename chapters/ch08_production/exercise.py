"""
第 8 章练习题

目标：给第 7 章的 Planner+Executor 系统加上生产化能力

要求：
  1. 【流式输出】用 app.stream() 逐步打印每个节点的输出，而不是等全部完成才显示
  2. 【异步】改写主函数为 async def main()，用 app.ainvoke() 调用
  3. 【降级】给 executor_node 中的 tool 调用加 try/except，
     工具失败时返回 "工具调用失败，跳过此步骤" 而不是抛出异常

  注意：不需要真正配置 LangSmith（需要付费账号），
        但要在代码顶部加上注释说明如何开启：
        # LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY in .env

验收标准：
  1. 运行时能看到逐节点的流式输出（不是最后一次性显示）
  2. 手动在某个 tool 里 raise Exception("test") 后程序不崩溃
  3. async main() 用 asyncio.run() 调用
"""
import asyncio
from dotenv import load_dotenv

load_dotenv()

# 如需开启 LangSmith tracing，在 .env 中设置：
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your_api_key
# LANGCHAIN_PROJECT=langchain-learning

# TODO 1: 从第 7 章复制 State、节点函数、graph 定义

# TODO 2: 改写 executor_node 中的 tool 调用，加 try/except 降级处理

# TODO 3: 定义 async def main()，用 app.ainvoke() 调用并打印结果

# TODO 4: 在 main() 之前，用 for chunk in app.stream(...) 打印流式输出

if __name__ == "__main__":
    asyncio.run(main())
