# 第 8 章：可观测性与生产化

## LangSmith Tracing

```bash
# .env 中设置
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=my-project
```

开启后，所有 LangChain 调用自动上报到 LangSmith，无需修改代码。
访问 https://smith.langchain.com 查看 trace 树。

## 流式输出

```python
# 同步流式
for chunk in chain.stream({"input": "hello"}):
    print(chunk, end="", flush=True)

# 异步流式
async for chunk in chain.astream({"input": "hello"}):
    print(chunk, end="", flush=True)
```

## 异步调用

```python
import asyncio

async def main():
    result = await chain.ainvoke({"input": "hello"})
    return result

asyncio.run(main())
```

## 优雅降级（Fallback）

```python
from langchain_core.runnables import RunnableLambda

def fallback_handler(error):
    return f"服务暂时不可用，请稍后重试。错误：{type(error).__name__}"

safe_chain = chain.with_fallbacks([RunnableLambda(fallback_handler)])
```

## 重试

```python
# 自动重试最多 3 次
retrying_chain = chain.with_retry(stop_after_attempt=3)
```
