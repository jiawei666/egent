# 第 1 章：LCEL —— LangChain 的组合语言

## 是什么

LCEL（LangChain Expression Language）是 LangChain 的核心组合方式，用 `|` 把组件串联成 pipeline。

所有 LangChain 组件都实现了 `Runnable` 接口，意味着它们都有：
- `.invoke(input)` —— 同步调用
- `.stream(input)` —— 流式输出
- `.batch([input1, input2])` —— 批量调用

## 为什么需要它

以前 LangChain 用 `LLMChain` 等类组合组件，耦合度高。LCEL 用函数式管道风格，每个组件独立可测，组合方式一致。

## 核心语法

```python
chain = prompt | model | parser  # 用 | 串联

# 三种调用方式
result = chain.invoke({"input": "hello"})
for chunk in chain.stream({"input": "hello"}):
    print(chunk, end="")
results = chain.batch([{"input": "hello"}, {"input": "world"}])
```

## RunnablePassthrough.assign

```python
from langchain_core.runnables import RunnablePassthrough

# 把原始输入 + 新字段一起传下去
chain = RunnablePassthrough.assign(definition=define_chain)
# 输入 {"word": "cat"} → 输出 {"word": "cat", "definition": "..."}
```
