# LangChain Agent 学习课程 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 8 章 LangChain Agent 学习路线创建完整的教学材料（concept.md、example.py、exercise.py、solution.py）

**Architecture:** 每章是独立目录 `chapters/chXX_<name>/`，包含 4 个文件，自成体系，可直接运行。章节之间在代码上互不依赖（练习题独立），但概念上前后承接。

**Tech Stack:** Python 3.14, langchain>=1.3.11, langchain-openai, langgraph, langsmith, python-dotenv

---

## 文件结构

```
egent/
  .env.example                          ← 环境变量模板（新建）
  pyproject.toml                        ← 添加新依赖（修改）
  chapters/
    ch01_lcel/
      concept.md                        ← 新建
      example.py                        ← 新建
      exercise.py                       ← 新建
      solution.py                       ← 新建
    ch02_prompt/
      concept.md / example.py / exercise.py / solution.py
    ch03_tools/
      concept.md / example.py / exercise.py / solution.py
    ch04_agent/
      concept.md / example.py / exercise.py / solution.py
    ch05_memory/
      concept.md / example.py / exercise.py / solution.py
    ch06_structured_output/
      concept.md / example.py / exercise.py / solution.py
    ch07_langgraph/
      concept.md / example.py / exercise.py / solution.py
    ch08_production/
      concept.md / example.py / exercise.py / solution.py
```

---

## Task 0: 环境配置

**Files:**
- Modify: `pyproject.toml`
- Create: `.env.example`

- [ ] **Step 1: 更新 pyproject.toml 添加依赖**

将 `pyproject.toml` 的 `dependencies` 改为：

```toml
[project]
name = "egent"
version = "0.1.0"
description = "LangChain Agent 学习项目"
readme = "README.md"
requires-python = ">=3.14"
dependencies = [
    "langchain>=1.3.11",
    "langchain-openai>=0.3.0",
    "langgraph>=0.2.0",
    "langsmith>=0.1.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.0.0",
]
```

- [ ] **Step 2: 安装依赖**

```bash
uv sync
```

期望输出：无报错，所有包安装成功。

- [ ] **Step 3: 创建 .env.example**

```bash
# .env.example
OPENAI_API_KEY=sk-...
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=                    # 第 8 章 LangSmith 才需要
LANGCHAIN_PROJECT=langchain-learning
```

- [ ] **Step 4: 创建本地 .env（不提交）**

```bash
cp .env.example .env
# 然后在 .env 中填入你的 OPENAI_API_KEY
```

- [ ] **Step 5: 创建章节目录**

```bash
mkdir -p chapters/ch01_lcel chapters/ch02_prompt chapters/ch03_tools \
         chapters/ch04_agent chapters/ch05_memory chapters/ch06_structured_output \
         chapters/ch07_langgraph chapters/ch08_production
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .env.example uv.lock chapters/
git commit -m "chore: setup curriculum project structure and dependencies"
```

---

## Task 1: 第 1 章 —— LCEL

**Files:**
- Create: `chapters/ch01_lcel/concept.md`
- Create: `chapters/ch01_lcel/example.py`
- Create: `chapters/ch01_lcel/exercise.py`
- Create: `chapters/ch01_lcel/solution.py`

- [ ] **Step 1: 创建 concept.md**

```markdown
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
```

- [ ] **Step 2: 创建 example.py**

```python
"""第 1 章示例：用 LCEL 构建一个翻译 pipeline"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

# Chain 1: 获取英文解释
define_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a dictionary. Give a concise English definition in one sentence."),
        ("human", "Word: {word}"),
    ])
    | model
    | parser
)

# Chain 2: 翻译成中文
translate_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Translate the following English text to Chinese. Only output the translation."),
        ("human", "{definition}"),
    ])
    | model
    | parser
)

# 组合 pipeline：word → definition → translation → 合并输出
pipeline = (
    RunnablePassthrough.assign(definition=define_chain)
    | RunnablePassthrough.assign(
        translation=lambda x: translate_chain.invoke({"definition": x["definition"]})
    )
    | (lambda x: f"英文解释: {x['definition']}\n中文翻译: {x['translation']}")
)

if __name__ == "__main__":
    result = pipeline.invoke({"word": "serendipity"})
    print(result)
```

- [ ] **Step 3: 运行 example.py 验证**

```bash
cd /Users/yuanjiawei/ai-coding/egent
uv run python chapters/ch01_lcel/example.py
```

期望输出（内容随模型不同略有差异）：
```
英文解释: The occurrence of events by chance in a happy or beneficial way.
中文翻译: 意外发现美好事物的能力。
```

- [ ] **Step 4: 创建 exercise.py**

```python
"""
第 1 章练习题

目标：用 LCEL 构建一个"单词卡"pipeline
  输入：{"word": "ephemeral"}
  输出：包含英文解释 + 中文翻译 + 一个例句 的格式化字符串

要求：
  1. 必须用 | 串联至少两个 LLM 调用步骤
  2. 最终输出格式：
     英文解释: ...
     中文翻译: ...
     例句: ...（英文）

验收标准：pipeline.invoke({"word": "ephemeral"}) 输出包含上述三个字段
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

# TODO 1: 创建 define_chain，输入 {"word": ...}，输出单词的英文解释（字符串）
define_chain = None  # 替换这行

# TODO 2: 创建 translate_chain，输入 {"definition": ...}，输出中文翻译（字符串）
translate_chain = None  # 替换这行

# TODO 3: 创建 example_chain，输入 {"word": ...}，输出一个英文例句（字符串）
example_chain = None  # 替换这行

# TODO 4: 用 RunnablePassthrough.assign 把三个 chain 组合成一个 pipeline
# 最终 lambda 把结果格式化成包含三个字段的字符串
pipeline = None  # 替换这行

if __name__ == "__main__":
    result = pipeline.invoke({"word": "ephemeral"})
    print(result)
    # 验收：输出包含"英文解释:"、"中文翻译:"、"例句:"三行
```

- [ ] **Step 5: 创建 solution.py**

```python
"""第 1 章练习题参考答案"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

define_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a dictionary. Give a concise English definition in one sentence."),
        ("human", "Word: {word}"),
    ])
    | model
    | parser
)

translate_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Translate the following English text to Chinese. Only output the translation."),
        ("human", "{definition}"),
    ])
    | model
    | parser
)

example_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Give one natural English example sentence using the word. Only the sentence."),
        ("human", "Word: {word}"),
    ])
    | model
    | parser
)

pipeline = (
    RunnablePassthrough.assign(definition=define_chain)
    | RunnablePassthrough.assign(
        translation=lambda x: translate_chain.invoke({"definition": x["definition"]})
    )
    | RunnablePassthrough.assign(example=example_chain)
    | (lambda x: f"英文解释: {x['definition']}\n中文翻译: {x['translation']}\n例句: {x['example']}")
)

if __name__ == "__main__":
    result = pipeline.invoke({"word": "ephemeral"})
    print(result)
```

- [ ] **Step 6: Commit**

```bash
git add chapters/ch01_lcel/
git commit -m "feat: add ch01 LCEL learning materials"
```

---

## Task 2: 第 2 章 —— Prompt + Model + OutputParser

**Files:**
- Create: `chapters/ch02_prompt/concept.md`
- Create: `chapters/ch02_prompt/example.py`
- Create: `chapters/ch02_prompt/exercise.py`
- Create: `chapters/ch02_prompt/solution.py`

- [ ] **Step 1: 创建 concept.md**

```markdown
# 第 2 章：Prompt + Model + OutputParser

## ChatPromptTemplate

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("human", "{question}"),
])

# .partial() 预填充部分变量
code_reviewer = prompt.partial(role="senior code reviewer")
```

## PydanticOutputParser

让 LLM 返回结构化数据，底层把 Pydantic schema 注入 prompt：

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class Review(BaseModel):
    issues: list[str] = Field(description="发现的问题列表")
    score: int = Field(description="代码质量评分 0-10")

parser = PydanticOutputParser(pydantic_object=Review)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Review the code.\n{format_instructions}"),
    ("human", "{code}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser
result = chain.invoke({"code": "def f(): return 1"})
print(result.score)   # 直接访问字段
print(result.issues)
```

## model.bind()

固定模型参数：

```python
cold_model = model.bind(temperature=0)
json_model = model.bind(response_format={"type": "json_object"})
```
```

- [ ] **Step 2: 创建 example.py**

```python
"""第 2 章示例：代码评审 bot，返回 Pydantic 结构体"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

class CodeReview(BaseModel):
    issues: list[str] = Field(description="List of issues found in the code")
    score: int = Field(description="Code quality score from 0 to 10")
    suggestion: str = Field(description="One key improvement suggestion")

parser = PydanticOutputParser(pydantic_object=CodeReview)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a senior Python code reviewer. Review the code and respond in the required format.\n{format_instructions}"),
    ("human", "Review this code:\n\n{code}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser

if __name__ == "__main__":
    sample_code = """
def calculate_average(numbers):
    total = 0
    for n in numbers:
        total = total + n
    return total / len(numbers)
"""
    result = chain.invoke({"code": sample_code})
    print(f"Score: {result.score}/10")
    print(f"Issues: {result.issues}")
    print(f"Suggestion: {result.suggestion}")
    # 验证字段可直接访问
    assert isinstance(result.score, int)
    assert isinstance(result.issues, list)
```

- [ ] **Step 3: 运行验证**

```bash
uv run python chapters/ch02_prompt/example.py
```

期望输出（内容因模型而异）：
```
Score: 6/10
Issues: ['No error handling for empty list (ZeroDivisionError)', 'Could use built-in sum()']
Suggestion: Use sum(numbers) / len(numbers) and add a guard for empty input.
```

- [ ] **Step 4: 创建 exercise.py**

```python
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
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
```

- [ ] **Step 5: 创建 solution.py**

```python
"""第 2 章练习题参考答案"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class DocResult(BaseModel):
    summary: str = Field(description="One sentence describing what the function does")
    params: list[str] = Field(description="Parameter descriptions in 'name: description' format")
    returns: str = Field(description="Description of the return value")
    example: str = Field(description="A usage example showing how to call the function")

parser = PydanticOutputParser(pydantic_object=DocResult)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a documentation writer. Generate clear documentation for the given Python function.\n{format_instructions}"),
    ("human", "Generate documentation for:\n\n{function_code}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser

if __name__ == "__main__":
    result = chain.invoke({"function_code": "def add(a, b):\n    return a + b"})
    print(f"Summary: {result.summary}")
    print(f"Params: {result.params}")
    print(f"Returns: {result.returns}")
    print(f"Example: {result.example}")
```

- [ ] **Step 6: Commit**

```bash
git add chapters/ch02_prompt/
git commit -m "feat: add ch02 Prompt+Model+Parser learning materials"
```

---

## Task 3: 第 3 章 —— Tools

**Files:**
- Create: `chapters/ch03_tools/concept.md`
- Create: `chapters/ch03_tools/example.py`
- Create: `chapters/ch03_tools/exercise.py`
- Create: `chapters/ch03_tools/solution.py`

- [ ] **Step 1: 创建 concept.md**

```markdown
# 第 3 章：Tools —— 给 LLM 加能力

## @tool 装饰器

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city. Returns temperature and conditions."""
    return f"{city}: 25°C, sunny"

# tool 的关键属性
print(get_weather.name)         # "get_weather"
print(get_weather.description)  # docstring 的内容
print(get_weather.args_schema)  # Pydantic schema，从类型注解自动生成

# 调用 tool
result = get_weather.invoke({"city": "Beijing"})
```

## 为什么 docstring 很重要

LLM 靠 `description` 决定什么时候调用这个 tool。
描述要清楚说明：tool 做什么、接受什么输入、返回什么格式。

差的描述："Gets weather"
好的描述："Get current weather for a city. Input: city name in English. Returns temperature in Celsius and weather conditions."

## 带多参数的 tool

```python
@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search for available flights. Date format: YYYY-MM-DD."""
    return f"Found 3 flights from {origin} to {destination} on {date}"
```
```

- [ ] **Step 2: 创建 example.py**

```python
"""第 3 章示例：定义和调用自定义 tools"""
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city. Returns temperature in Celsius and conditions.
    Use this when the user asks about weather in a specific location."""
    weather_data = {
        "Beijing": "12°C, cloudy",
        "Shanghai": "18°C, sunny",
        "Shenzhen": "26°C, partly cloudy",
    }
    return weather_data.get(city, f"{city}: 20°C, unknown conditions")

@tool
def get_time(timezone: str) -> str:
    """Get current time for a timezone. Accepts timezone names like 'Asia/Shanghai', 'UTC', 'US/Eastern'.
    Use this when the user asks what time it is in a specific place."""
    from datetime import datetime
    import zoneinfo
    try:
        tz = zoneinfo.ZoneInfo(timezone)
        now = datetime.now(tz)
        return f"Current time in {timezone}: {now.strftime('%H:%M:%S %Z')}"
    except Exception:
        return f"Unknown timezone: {timezone}"

if __name__ == "__main__":
    # 直接调用 tool（不经过 LLM）
    print(get_weather.invoke({"city": "Beijing"}))
    print(get_time.invoke({"timezone": "Asia/Shanghai"}))

    # 查看 tool 的元数据（LLM 靠这些决定何时调用）
    print(f"\nTool name: {get_weather.name}")
    print(f"Tool description: {get_weather.description}")
    print(f"Args schema: {get_weather.args_schema.model_json_schema()}")
```

- [ ] **Step 3: 运行验证**

```bash
uv run python chapters/ch03_tools/example.py
```

期望输出：
```
Beijing: 12°C, cloudy
Current time in Asia/Shanghai: 14:32:10 CST

Tool name: get_weather
Tool description: Get current weather for a city. ...
Args schema: {'properties': {'city': {'title': 'City', 'type': 'string'}}, ...}
```

- [ ] **Step 4: 创建 exercise.py**

```python
"""
第 3 章练习题

目标：创建两个自定义 tools
  1. search_news(topic: str, max_results: int = 3) -> str
     返回格式："{max_results} news articles about {topic}: [article1, article2, ...]"
     用硬编码假数据即可

  2. calculate_exchange(amount: float, from_currency: str, to_currency: str) -> str
     支持 USD/CNY/EUR 之间的换算（用固定汇率即可）
     返回格式："100.0 USD = 725.0 CNY"

要求：
  1. docstring 要清楚描述输入输出格式
  2. tool.invoke({"topic": "AI", "max_results": 2}) 能正确触发
  3. 打印每个 tool 的 name、description、args_schema

验收标准：
  - tool.name 是 "search_news" 和 "calculate_exchange"
  - tool.description 包含参数和返回格式说明
  - tool.invoke(...) 返回预期格式的字符串
"""
from langchain_core.tools import tool

# TODO 1: 定义 search_news tool
# 参数：topic: str, max_results: int = 3
# 用硬编码新闻数据，返回 f"{max_results} news articles about {topic}: ..."


# TODO 2: 定义 calculate_exchange tool
# 参数：amount: float, from_currency: str, to_currency: str
# 汇率：1 USD = 7.25 CNY, 1 USD = 0.92 EUR
# 返回格式：f"{amount} {from_currency} = {result} {to_currency}"


if __name__ == "__main__":
    # 测试 search_news
    print(search_news.invoke({"topic": "artificial intelligence", "max_results": 2}))
    print(f"Name: {search_news.name}")
    print(f"Description: {search_news.description}\n")

    # 测试 calculate_exchange
    print(calculate_exchange.invoke({"amount": 100.0, "from_currency": "USD", "to_currency": "CNY"}))
    print(f"Name: {calculate_exchange.name}")
    print(f"Description: {calculate_exchange.description}")
```

- [ ] **Step 5: 创建 solution.py**

```python
"""第 3 章练习题参考答案"""
from langchain_core.tools import tool

@tool
def search_news(topic: str, max_results: int = 3) -> str:
    """Search for recent news articles on a topic.
    Input: topic (search query string), max_results (number of articles, default 3).
    Returns: a summary string listing article headlines."""
    headlines = [
        f"Breaking: Major {topic} development announced",
        f"Experts weigh in on {topic} trends",
        f"New research reveals {topic} insights",
        f"Government responds to {topic} concerns",
        f"Industry leaders discuss {topic} future",
    ]
    selected = headlines[:max_results]
    return f"{max_results} news articles about '{topic}': {'; '.join(selected)}"

@tool
def calculate_exchange(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between currencies (USD, CNY, EUR).
    Input: amount (float), from_currency (e.g. 'USD'), to_currency (e.g. 'CNY').
    Returns: conversion result string like '100.0 USD = 725.0 CNY'."""
    rates_to_usd = {"USD": 1.0, "CNY": 1 / 7.25, "EUR": 1 / 0.92}
    if from_currency not in rates_to_usd or to_currency not in rates_to_usd:
        return f"Unsupported currency. Supported: USD, CNY, EUR"
    amount_in_usd = amount * rates_to_usd[from_currency]
    result = amount_in_usd / rates_to_usd[to_currency]
    return f"{amount} {from_currency} = {result:.2f} {to_currency}"

if __name__ == "__main__":
    print(search_news.invoke({"topic": "artificial intelligence", "max_results": 2}))
    print(f"Name: {search_news.name}")
    print(f"Description: {search_news.description}\n")

    print(calculate_exchange.invoke({"amount": 100.0, "from_currency": "USD", "to_currency": "CNY"}))
    print(f"Name: {calculate_exchange.name}")
    print(f"Description: {calculate_exchange.description}")
```

- [ ] **Step 6: Commit**

```bash
git add chapters/ch03_tools/
git commit -m "feat: add ch03 Tools learning materials"
```

---

## Task 4: 第 4 章 —— 第一个 Agent（ReAct）

**Files:**
- Create: `chapters/ch04_agent/concept.md`
- Create: `chapters/ch04_agent/example.py`
- Create: `chapters/ch04_agent/exercise.py`
- Create: `chapters/ch04_agent/solution.py`

- [ ] **Step 1: 创建 concept.md**

```markdown
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
```

- [ ] **Step 2: 创建 example.py**

```python
"""第 4 章示例：第一个 ReAct Agent"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city. Returns temperature in Celsius and conditions."""
    data = {"Beijing": "12°C, cloudy", "Shanghai": "18°C, sunny", "Shenzhen": "26°C, partly cloudy"}
    return data.get(city, f"{city}: data not available")

@tool
def get_time(timezone: str) -> str:
    """Get current local time for a timezone (e.g. 'Asia/Shanghai', 'UTC')."""
    from datetime import datetime
    import zoneinfo
    try:
        now = datetime.now(zoneinfo.ZoneInfo(timezone))
        return f"Current time in {timezone}: {now.strftime('%H:%M:%S %Z')}"
    except Exception:
        return f"Unknown timezone: {timezone}"

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [get_weather, get_time]

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(model, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
)

if __name__ == "__main__":
    result = executor.invoke({"input": "What's the current time in Beijing and what's the weather like there?"})
    print("\n=== Final Answer ===")
    print(result["output"])
```

- [ ] **Step 3: 运行验证**

```bash
uv run python chapters/ch04_agent/example.py
```

期望输出：可以看到 `Thought:` / `Action:` / `Observation:` 的 ReAct 循环，最终打印 `Final Answer`。
验证：Agent 分别调用了 `get_time` 和 `get_weather` 两个 tool（不是直接猜测答案）。

- [ ] **Step 4: 创建 exercise.py**

```python
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
```

- [ ] **Step 5: 创建 solution.py**

```python
"""第 4 章练习题参考答案"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city. Returns temperature in Celsius and conditions."""
    data = {"Beijing": "12°C, cloudy", "Shanghai": "18°C, sunny", "Shenzhen": "26°C, partly cloudy"}
    return data.get(city, f"{city}: data not available")

@tool
def get_time(timezone: str) -> str:
    """Get current local time for a timezone (e.g. 'Asia/Shanghai', 'UTC')."""
    from datetime import datetime
    import zoneinfo
    try:
        now = datetime.now(zoneinfo.ZoneInfo(timezone))
        return f"Current time in {timezone}: {now.strftime('%H:%M:%S %Z')}"
    except Exception:
        return f"Unknown timezone: {timezone}"

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [get_weather, get_time]
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(model, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5, handle_parsing_errors=True)

if __name__ == "__main__":
    result = executor.invoke({"input": "北京现在几点？天气怎么样？"})
    print("\n=== Final Answer ===")
    print(result["output"])
```

- [ ] **Step 6: Commit**

```bash
git add chapters/ch04_agent/
git commit -m "feat: add ch04 ReAct Agent learning materials"
```

---

## Task 5: 第 5 章 —— Memory

**Files:**
- Create: `chapters/ch05_memory/concept.md`
- Create: `chapters/ch05_memory/example.py`
- Create: `chapters/ch05_memory/exercise.py`
- Create: `chapters/ch05_memory/solution.py`

- [ ] **Step 1: 创建 concept.md**

```markdown
# 第 5 章：Memory —— 让 Agent 记住对话

## 为什么需要 Memory

默认 AgentExecutor 每次调用都是全新的——不记得上一轮说了什么。
Memory 把历史消息注入 prompt，让 Agent 感知多轮上下文。

## RunnableWithMessageHistory

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}  # session_id -> ChatMessageHistory

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

agent_with_history = RunnableWithMessageHistory(
    executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# 同一 session_id 的调用会共享历史
config = {"configurable": {"session_id": "user_123"}}
agent_with_history.invoke({"input": "我叫小明"}, config=config)
agent_with_history.invoke({"input": "我叫什么名字？"}, config=config)
```

## 关键参数

- `input_messages_key`：本次用户输入的 key（对应 prompt 中的变量名）
- `history_messages_key`：历史消息注入 prompt 的 key（prompt 中必须有 `{chat_history}` 占位符）
- `session_id`：区分不同用户/会话，不同 session 的历史互不影响
```

- [ ] **Step 2: 创建 example.py**

```python
"""第 5 章示例：给 Agent 加上多轮对话记忆"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    data = {"Beijing": "12°C, cloudy", "Shanghai": "18°C, sunny"}
    return data.get(city, f"{city}: data not available")

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [get_weather]

# 使用支持 chat_history 的 prompt（create_openai_tools_agent 用于 tool calling 模型）
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(model, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

agent_with_history = RunnableWithMessageHistory(
    executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

if __name__ == "__main__":
    config = {"configurable": {"session_id": "demo_session"}}

    r1 = agent_with_history.invoke({"input": "帮我查一下北京的天气"}, config=config)
    print(f"第一轮: {r1['output']}")

    r2 = agent_with_history.invoke({"input": "刚才你查的是哪个城市？"}, config=config)
    print(f"第二轮: {r2['output']}")
    # 验收：第二轮回答中包含 "Beijing" 或 "北京"
    assert "beijing" in r2["output"].lower() or "北京" in r2["output"], \
        "Agent 没能记住上一轮的城市！"
    print("✓ Memory 正常工作")
```

- [ ] **Step 3: 运行验证**

```bash
uv run python chapters/ch05_memory/example.py
```

期望输出：
```
第一轮: 北京当前天气是 12°C，多云。
第二轮: 刚才我查询的是北京（Beijing）的天气。
✓ Memory 正常工作
```

- [ ] **Step 4: 创建 exercise.py**

```python
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
```

- [ ] **Step 5: 创建 solution.py**

```python
"""第 5 章练习题参考答案"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city. Returns temperature and conditions."""
    data = {"Beijing": "12°C, cloudy", "Shanghai": "28°C, sunny", "Harbin": "−5°C, snowy"}
    return data.get(city, f"{city}: data not available")

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [get_weather]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Remember the user's preferences throughout the conversation."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(model, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

agent_with_history = RunnableWithMessageHistory(
    executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

if __name__ == "__main__":
    config = {"configurable": {"session_id": "travel_session"}}

    r1 = agent_with_history.invoke({"input": "我想去北京旅行，我不喜欢太热的天气"}, config=config)
    print(f"第一轮: {r1['output']}\n")

    r2 = agent_with_history.invoke({"input": "根据我的喜好，北京现在适合去吗？"}, config=config)
    print(f"第二轮: {r2['output']}\n")

    r3 = agent_with_history.invoke({"input": "我之前说不喜欢什么样的天气？"}, config=config)
    print(f"第三轮: {r3['output']}")
    assert "热" in r3["output"] or "hot" in r3["output"].lower(), "Memory 失效！"
    print("✓ Memory 正常工作")
```

- [ ] **Step 6: Commit**

```bash
git add chapters/ch05_memory/
git commit -m "feat: add ch05 Memory learning materials"
```

---

## Task 6: 第 6 章 —— Structured Output

**Files:**
- Create: `chapters/ch06_structured_output/concept.md`
- Create: `chapters/ch06_structured_output/example.py`
- Create: `chapters/ch06_structured_output/exercise.py`
- Create: `chapters/ch06_structured_output/solution.py`

- [ ] **Step 1: 创建 concept.md**

```markdown
# 第 6 章：Structured Output —— Agent 返回结构化数据

## with_structured_output

让 LLM 返回 Pydantic 对象，底层用 function calling 实现（比 PydanticOutputParser 更可靠）：

```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class Person(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age")
    city: str = Field(description="City where person lives")

model = ChatOpenAI(model="gpt-4o-mini")
structured_model = model.with_structured_output(Person)

result = structured_model.invoke("John is 30 years old and lives in New York.")
print(result.name)  # "John"
print(result.age)   # 30
```

## 与 PydanticOutputParser 的区别

| | PydanticOutputParser | with_structured_output |
|---|---|---|
| 底层机制 | 让 LLM 输出 JSON 字符串，再 parse | Function calling，LLM 直接填结构体 |
| 可靠性 | LLM 可能输出格式错误 | 更可靠，OpenAI 原生支持 |
| 适用场景 | 所有模型 | 支持 function calling 的模型 |

## 在 LCEL chain 中使用

```python
chain = prompt | model.with_structured_output(MySchema)
result = chain.invoke({"text": "..."})
# result 直接是 MySchema 实例
```
```

- [ ] **Step 2: 创建 example.py**

```python
"""第 6 章示例：信息提取，返回结构化 Pydantic 对象"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

class NewsInfo(BaseModel):
    persons: list[str] = Field(description="Names of people mentioned in the news")
    location: str = Field(description="Primary location where the event occurred")
    time: str = Field(description="When the event occurred (date or time period)")
    event_summary: str = Field(description="One sentence summary of what happened")

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_model = model.with_structured_output(NewsInfo)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract structured information from the news article."),
    ("human", "{text}"),
])

chain = prompt | structured_model

if __name__ == "__main__":
    news = """
    Apple CEO Tim Cook announced on Tuesday in Cupertino, California, that the company
    will invest $500 billion in the United States over the next four years.
    CFO Luca Maestri also attended the press conference alongside Cook.
    """
    result = chain.invoke({"text": news})
    print(f"Persons: {result.persons}")
    print(f"Location: {result.location}")
    print(f"Time: {result.time}")
    print(f"Event: {result.event_summary}")
    assert isinstance(result.persons, list)
    assert isinstance(result.event_summary, str)
    print("✓ Structured output 正常工作")
```

- [ ] **Step 3: 运行验证**

```bash
uv run python chapters/ch06_structured_output/example.py
```

期望输出：
```
Persons: ['Tim Cook', 'Luca Maestri']
Location: Cupertino, California
Time: Tuesday
Event: Apple announced a $500 billion investment in the United States over the next four years.
✓ Structured output 正常工作
```

- [ ] **Step 4: 创建 exercise.py**

```python
"""
第 6 章练习题

目标：构建一个"简历解析器"
  输入：一段简历文本
  输出：ResumeInfo 对象，包含：
    - name: str（姓名）
    - skills: list[str]（技能列表）
    - years_of_experience: int（工作年限，无法判断时填 0）
    - education: str（最高学历）
    - summary: str（一句话简介）

要求：
  1. 用 model.with_structured_output(ResumeInfo)
  2. result.skills 必须是 list，不是字符串
  3. 用 ChatPromptTemplate 构建 prompt

验收标准：
  对下面的测试简历，result.name 包含"张伟"，result.skills 是非空列表
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# TODO 1: 定义 ResumeInfo Pydantic 模型（5 个字段）

# TODO 2: 创建 structured_model = model.with_structured_output(ResumeInfo)

# TODO 3: 创建 prompt 和 chain

if __name__ == "__main__":
    resume = """
    张伟，5 年 Python 后端开发经验，熟练掌握 FastAPI、Django、PostgreSQL、Redis、Docker。
    本科毕业于北京大学计算机系，曾在字节跳动和阿里巴巴任职。
    目前专注于 AI 应用开发，有 LangChain 和 OpenAI API 使用经验。
    """
    # TODO 4: result = chain.invoke({"text": resume})
    # 打印所有字段并验证 name 包含"张伟"，skills 是非空列表
    pass
```

- [ ] **Step 5: 创建 solution.py**

```python
"""第 6 章练习题参考答案"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class ResumeInfo(BaseModel):
    name: str = Field(description="Full name of the candidate")
    skills: list[str] = Field(description="List of technical skills mentioned")
    years_of_experience: int = Field(description="Total years of work experience, 0 if unknown")
    education: str = Field(description="Highest level of education and institution")
    summary: str = Field(description="One sentence professional summary")

structured_model = model.with_structured_output(ResumeInfo)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a resume parser. Extract structured information from the resume text."),
    ("human", "{text}"),
])

chain = prompt | structured_model

if __name__ == "__main__":
    resume = """
    张伟，5 年 Python 后端开发经验，熟练掌握 FastAPI、Django、PostgreSQL、Redis、Docker。
    本科毕业于北京大学计算机系，曾在字节跳动和阿里巴巴任职。
    目前专注于 AI 应用开发，有 LangChain 和 OpenAI API 使用经验。
    """
    result = chain.invoke({"text": resume})
    print(f"Name: {result.name}")
    print(f"Skills: {result.skills}")
    print(f"Experience: {result.years_of_experience} years")
    print(f"Education: {result.education}")
    print(f"Summary: {result.summary}")
    assert "张伟" in result.name
    assert isinstance(result.skills, list) and len(result.skills) > 0
    print("✓ Resume parser 正常工作")
```

- [ ] **Step 6: Commit**

```bash
git add chapters/ch06_structured_output/
git commit -m "feat: add ch06 Structured Output learning materials"
```

---

## Task 7: 第 7 章 —— LangGraph

**Files:**
- Create: `chapters/ch07_langgraph/concept.md`
- Create: `chapters/ch07_langgraph/example.py`
- Create: `chapters/ch07_langgraph/exercise.py`
- Create: `chapters/ch07_langgraph/solution.py`

- [ ] **Step 1: 创建 concept.md**

```markdown
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
```

- [ ] **Step 2: 创建 example.py**

```python
"""第 7 章示例：Planner + Executor 双 Agent 系统"""
from dotenv import load_dotenv
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
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
    result = app.invoke({"task": "Research the top 3 Python web frameworks and their key features"})
    print("\n=== Final Answer ===")
    print(result["final_answer"])
```

- [ ] **Step 3: 运行验证**

```bash
uv run python chapters/ch07_langgraph/example.py
```

期望输出：可以看到 `[Planner]` 输出步骤列表，`[Executor]` 逐步执行，最终打印完整总结。

- [ ] **Step 4: 创建 exercise.py**

```python
"""
第 7 章练习题

目标：构建一个"写作助手" LangGraph 系统
  节点：
    1. outliner —— 给定主题，生成文章大纲（3-4 个要点）
    2. writer —— 逐点展开写作内容
    3. reviewer —— 审阅全文，给出修改建议或通过

  流程：
    START → outliner → writer（循环，每次写一个要点）→ reviewer → END
    如果 reviewer 认为需要修改（state["needs_revision"] = True），回到 writer 重写最后一个要点；
    否则结束。

State 结构：
  - topic: str
  - outline: list[str]
  - current_point: int
  - sections: list[str]（每个要点的写作内容）
  - needs_revision: bool
  - revision_note: str
  - final_article: str

验收标准：
  app.invoke({"topic": "Python 异步编程入门"}) 能输出一篇结构完整的文章
"""
from dotenv import load_dotenv
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

class State(TypedDict):
    topic: str
    outline: list[str]
    current_point: int
    sections: list[str]
    needs_revision: bool
    revision_note: str
    final_article: str

# TODO 1: 定义 outliner_node（输入 topic，输出 outline 列表和 current_point=0）

# TODO 2: 定义 writer_node（读取 outline[current_point]，写一段内容，追加到 sections，current_point+1）

# TODO 3: 定义 reviewer_node（读取所有 sections，判断质量，返回 needs_revision 和 revision_note）
#          如果通过，把 sections 合并成 final_article

# TODO 4: 定义路由函数
#   writer_router: current_point < len(outline) → "write"，否则 → "review"
#   reviewer_router: needs_revision → "revise"（回 writer），否则 → "finish"（到 END）

# TODO 5: 构建 StateGraph，编译，运行

if __name__ == "__main__":
    pass  # 替换为 app.invoke({"topic": "Python 异步编程入门", ...})
```

- [ ] **Step 5: 创建 solution.py**

```python
"""第 7 章练习题参考答案"""
from dotenv import load_dotenv
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

class State(TypedDict):
    topic: str
    outline: list[str]
    current_point: int
    sections: list[str]
    needs_revision: bool
    revision_note: str
    final_article: str

outline_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are an article planner. Create a 3-4 point outline for an article. Return ONLY a numbered list."),
        ("human", "Topic: {topic}"),
    ]) | model | parser
)

write_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a technical writer. Write a focused 2-3 paragraph section for the given outline point."),
        ("human", "Topic: {topic}\nSection to write: {point}\nRevision note (if any): {note}"),
    ]) | model | parser
)

review_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are an editor. Review the article draft. If it's good enough, say 'APPROVED'. If it needs work, say 'REVISE: ' followed by specific feedback."),
        ("human", "Topic: {topic}\nDraft:\n{draft}"),
    ]) | model | parser
)

def outliner_node(state: State) -> dict:
    print(f"\n[Outliner] Creating outline for: {state['topic']}")
    text = outline_chain.invoke({"topic": state["topic"]})
    points = [l.split(". ", 1)[1] if ". " in l else l for l in text.split("\n") if l.strip() and l.strip()[0].isdigit()]
    print(f"[Outliner] Outline: {points}")
    return {"outline": points, "current_point": 0, "sections": [], "needs_revision": False, "revision_note": ""}

def writer_node(state: State) -> dict:
    point = state["outline"][state["current_point"]]
    print(f"\n[Writer] Writing section: {point}")
    content = write_chain.invoke({"topic": state["topic"], "point": point, "note": state.get("revision_note", "")})
    updated_sections = list(state["sections"])
    if state.get("needs_revision") and updated_sections:
        updated_sections[-1] = content
    else:
        updated_sections.append(content)
    return {"sections": updated_sections, "current_point": state["current_point"] + 1, "needs_revision": False, "revision_note": ""}

def reviewer_node(state: State) -> dict:
    print(f"\n[Reviewer] Reviewing article...")
    draft = "\n\n".join(state["sections"])
    review = review_chain.invoke({"topic": state["topic"], "draft": draft})
    if "APPROVED" in review:
        print("[Reviewer] APPROVED")
        title = f"# {state['topic']}\n\n"
        sections_with_headers = [f"## {state['outline'][i]}\n\n{s}" for i, s in enumerate(state["sections"])]
        return {"needs_revision": False, "final_article": title + "\n\n".join(sections_with_headers)}
    note = review.replace("REVISE:", "").strip()
    print(f"[Reviewer] Needs revision: {note}")
    return {"needs_revision": True, "revision_note": note, "current_point": len(state["sections"]) - 1}

def writer_router(state: State) -> str:
    if state["current_point"] < len(state["outline"]):
        return "write"
    return "review"

def reviewer_router(state: State) -> str:
    return "revise" if state.get("needs_revision") else "finish"

graph = StateGraph(State)
graph.add_node("outliner", outliner_node)
graph.add_node("writer", writer_node)
graph.add_node("reviewer", reviewer_node)
graph.add_edge(START, "outliner")
graph.add_edge("outliner", "writer")
graph.add_conditional_edges("writer", writer_router, {"write": "writer", "review": "reviewer"})
graph.add_conditional_edges("reviewer", reviewer_router, {"revise": "writer", "finish": END})

app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({"topic": "Python 异步编程入门", "outline": [], "current_point": 0, "sections": [], "needs_revision": False, "revision_note": "", "final_article": ""})
    print("\n=== Final Article ===")
    print(result["final_article"])
```

- [ ] **Step 6: Commit**

```bash
git add chapters/ch07_langgraph/
git commit -m "feat: add ch07 LangGraph learning materials"
```

---

## Task 8: 第 8 章 —— 可观测性与生产化

**Files:**
- Create: `chapters/ch08_production/concept.md`
- Create: `chapters/ch08_production/example.py`
- Create: `chapters/ch08_production/exercise.py`
- Create: `chapters/ch08_production/solution.py`

- [ ] **Step 1: 创建 concept.md**

```markdown
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
```

- [ ] **Step 2: 创建 example.py**

```python
"""第 8 章示例：流式输出 + 异步 + 优雅降级"""
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

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

    @tool
    def unreliable_tool(query: str) -> str:
        """A tool that sometimes fails."""
        raise ConnectionError("模拟网络超时")

    def handle_tool_error(error: Exception) -> str:
        return f"工具调用失败（{type(error).__name__}），已使用默认回答：暂无相关数据。"

    safe_chain = RunnableLambda(lambda x: unreliable_tool.invoke(x)).with_fallbacks(
        [RunnableLambda(lambda x: handle_tool_error(ConnectionError("模拟错误")))]
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
```

- [ ] **Step 3: 运行验证**

```bash
uv run python chapters/ch08_production/example.py
```

期望输出：流式逐字输出、异步结果、降级提示，最终打印 `✓ 所有生产化功能正常工作`。

- [ ] **Step 4: 创建 exercise.py**

```python
"""
第 8 章练习题

目标：给第 7 章的 Planner+Executor 系统加上生产化能力

要求：
  1. 【流式输出】用 app.stream() 逐步打印每个节点的输出，而不是等全部完成才显示
  2. 【异步】改写主函数为 async def main()，用 app.ainvoke() 调用
  3. 【降级】给 executor_node 中的 tool 调用加 try/except，
     工具失败时返回 "工具调用失败，跳过此步骤" 而不是抛出异常

  注意：不需要真正配置 LangSmith（需要付费账号），
        但要在代码顶部加上检查 LANGCHAIN_TRACING_V2 的注释说明如何开启

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
```

- [ ] **Step 5: 创建 solution.py**

```python
"""第 8 章练习题参考答案"""
import asyncio
from dotenv import load_dotenv
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

load_dotenv()

# 如需开启 LangSmith tracing，在 .env 中设置：
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your_api_key
# LANGCHAIN_PROJECT=langchain-learning

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

class State(TypedDict):
    task: str
    plan: list[str]
    current_step: int
    results: list[str]
    final_answer: str

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    data = {
        "Python web frameworks": "Top: FastAPI, Django, Flask",
        "FastAPI": "Fast, async, auto-docs, Pydantic",
        "Django": "Batteries-included, ORM, admin",
        "Flask": "Minimal, flexible, easy to learn",
    }
    for k, v in data.items():
        if k.lower() in query.lower():
            return v
    return f"General info about: {query}"

planner_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Break down the task into 3 concrete steps. Return ONLY a numbered list."),
        ("human", "Task: {task}"),
    ]) | model | parser
)

def planner_node(state: State) -> dict:
    print(f"\n[Planner] Planning: {state['task']}")
    text = planner_chain.invoke({"task": state["task"]})
    steps = [l.split(". ", 1)[1] if ". " in l else l for l in text.split("\n") if l.strip() and l.strip()[0].isdigit()]
    return {"plan": steps, "current_step": 0, "results": []}

def executor_node(state: State) -> dict:
    step = state["plan"][state["current_step"]]
    print(f"\n[Executor] Step {state['current_step'] + 1}: {step}")
    try:
        result = search_web.invoke({"query": step})
    except Exception as e:
        result = f"工具调用失败（{type(e).__name__}），跳过此步骤"
    return {
        "results": state["results"] + [f"Step {state['current_step'] + 1}: {result}"],
        "current_step": state["current_step"] + 1,
    }

def finalizer_node(state: State) -> dict:
    summary_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "Summarize the research into a clear answer."),
            ("human", "Task: {task}\nResults:\n{results}"),
        ]) | model | parser
    )
    final = summary_chain.invoke({"task": state["task"], "results": "\n".join(state["results"])})
    return {"final_answer": final}

def should_continue(state: State) -> str:
    return "execute" if state["current_step"] < len(state["plan"]) else "finalize"

graph = StateGraph(State)
graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("finalizer", finalizer_node)
graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", should_continue, {"execute": "executor", "finalize": "finalizer"})
graph.add_edge("finalizer", END)
app = graph.compile()

async def main():
    print("=== 流式输出（每个节点完成时显示）===")
    async for chunk in app.astream(
        {"task": "Research Python web frameworks", "plan": [], "current_step": 0, "results": [], "final_answer": ""}
    ):
        node_name = list(chunk.keys())[0]
        print(f"[节点 {node_name} 完成]")

    print("\n=== 异步完整调用 ===")
    result = await app.ainvoke(
        {"task": "Research Python web frameworks", "plan": [], "current_step": 0, "results": [], "final_answer": ""}
    )
    print("\n=== Final Answer ===")
    print(result["final_answer"])

if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 6: Commit**

```bash
git add chapters/ch08_production/
git commit -m "feat: add ch08 Production learning materials"
```

---

## Self-Review

**Spec coverage check:**
- ✓ 第 1 章 LCEL：concept.md + example + exercise + solution
- ✓ 第 2 章 Prompt+Model+Parser：完整四文件
- ✓ 第 3 章 Tools：完整四文件
- ✓ 第 4 章 ReAct Agent：完整四文件
- ✓ 第 5 章 Memory：完整四文件
- ✓ 第 6 章 Structured Output：完整四文件
- ✓ 第 7 章 LangGraph：完整四文件
- ✓ 第 8 章 Production：完整四文件
- ✓ 环境配置 Task 0：pyproject.toml + .env.example

**Placeholder scan:** 无 TBD/TODO 残留（exercise.py 中的 TODO 是给学生的，不是设计缺口）。

**Type consistency:** 所有章节的 State TypedDict 字段、tool 参数类型、Pydantic 模型字段在 example.py 和 solution.py 之间保持一致。第 7-8 章 solution.py 中 State 初始化时所有字段显式提供。
