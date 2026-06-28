# LangChain Agent 学习路线设计文档

**日期：** 2026-06-28  
**目标：** 从「有 OpenAI API + LangChain 基础」到「能独立构建生产级 Agent 系统」

---

## 用户背景

- Python 熟练
- 有 OpenAI API 调用经验，了解 prompt/completion 基本概念
- 用过 LangChain 基础功能，但缺乏系统认识
- 最终目标：构建 AI Agent（自主使用工具、完成多步任务）
- 学习方式：边学边做，每个概念配一道练习题，自主掌握节奏

---

## 学习原则

- 每章 = 1 个核心抽象 + 1 道练习题
- 练习题有明确验收标准（跑通 + 输出符合预期），做完才进下一章
- 每章提供：概念说明、最小可运行示例、练习题（含 TODO）、参考答案

---

## 总体结构

```
阶段一：打地基（第 1-3 章）
  第 1 章  LCEL —— LangChain 的组合语言
  第 2 章  Prompt + Model + OutputParser
  第 3 章  Tools —— 给 LLM 加工具

阶段二：构建 Agent（第 4-6 章）
  第 4 章  第一个 Agent（ReAct 模式）
  第 5 章  Memory —— 让 Agent 记住对话
  第 6 章  Structured Output —— Agent 返回结构化数据

阶段三：进阶与生产（第 7-8 章）
  第 7 章  LangGraph —— 多 Agent 协作
  第 8 章  可观测性与生产化
```

---

## 各章详细设计

### 第 1 章：LCEL —— LangChain 的组合语言

**概念：** LCEL（LangChain Expression Language）是 LangChain 的核心，用 `|` 把任意 Runnable 串联成 pipeline。理解它是后续一切的基础。

**关键点：**
- `Runnable` 接口统一了所有组件的调用方式
- `|` 操作符串联组件
- `.invoke()` / `.stream()` / `.batch()` 三种调用方式

**练习题：** 用 LCEL 构建一个 pipeline，输入一个英文单词，输出该单词的英文解释 + 中文翻译（两步串联）。

**验收标准：** `.invoke("serendipity")` 返回包含解释和翻译的字符串，且两步是通过 `|` 串联完成的，不是一次 prompt 搞定的。

---

### 第 2 章：Prompt + Model + OutputParser

**概念：** LangChain 三件套。`ChatPromptTemplate` 管模板，`ChatOpenAI` 管调用，`StrOutputParser` / `PydanticOutputParser` 管解析输出格式。

**关键点：**
- `ChatPromptTemplate.from_messages()` 构建多角色 prompt
- `.partial()` 预填充部分参数
- `.bind()` 绑定模型参数（如 temperature）

**练习题：** 构建一个「代码评审 bot」——输入代码片段，输出包含 `issues`（问题列表）和 `score`（0-10 分）字段的 Pydantic 对象。

**验收标准：** 返回对象可直接 `.issues` 和 `.score` 访问，不需要 parse 字符串。

---

### 第 3 章：Tools —— 给 LLM 加能力

**概念：** Tool 是 LLM 能调用的函数。LangChain 有内置 tool（搜索、计算器），也支持用 `@tool` 装饰器自定义。Tool 的 docstring 即是 LLM 看到的描述，schema 自动生成。

**关键点：**
- `@tool` 装饰器
- docstring 的质量直接影响 LLM 调用准确性
- tool schema 由类型注解自动生成

**练习题：** 写两个自定义 tool：`get_weather(city: str)` 和 `get_time(timezone: str)`，用硬编码假数据返回。

**验收标准：** `tool.invoke({"city": "Beijing"})` 能正确触发并返回结果；`tool.name` 和 `tool.description` 字段内容清晰。

---

### 第 4 章：第一个 Agent（ReAct 模式）

**概念：** Agent = LLM + Tools + 循环推理。ReAct 是最经典的模式：Reasoning（思考下一步）→ Acting（调用 tool）→ Observing（看结果）→ 循环直到任务完成。

**关键点：**
- `create_react_agent` 创建 agent
- `AgentExecutor` 负责运行循环
- `agent_scratchpad` 存储中间推理过程

**练习题：** 用第 3 章的 `get_weather` 和 `get_time` 两个 tool 构建一个 Agent，让它回答「北京现在几点，天气怎么样？」

**验收标准：** Agent 能自动调用两个 tool 并综合回答，通过 `verbose=True` 可以看到它分别调用了两次 tool，而不是靠 LLM 直接猜测。

---

### 第 5 章：Memory —— 让 Agent 记住对话

**概念：** 默认 Agent 无状态，每次调用互不相知。Memory 把历史消息注入 prompt，让 Agent 感知多轮上下文。

**关键点：**
- `RunnableWithMessageHistory` 包装 agent
- `InMemoryChatMessageHistory` 存储历史
- `session_id` 区分不同会话

**练习题：** 给第 4 章的 Agent 加上 Memory，进行多轮对话：第一轮问北京天气，第二轮问「刚才你查的是哪个城市？」

**验收标准：** Agent 第二轮能正确回忆第一轮查询的城市，不需要用户重复说明。

---

### 第 6 章：Structured Output —— Agent 返回结构化数据

**概念：** 用 `with_structured_output` + Pydantic 让 LLM 不只返回自然语言，而是返回可编程处理的结构体，底层基于 function calling。

**关键点：**
- `model.with_structured_output(Schema)`
- Pydantic 模型定义输出结构
- `tool_choice` 强制调用

**练习题：** 构建一个「信息提取 Agent」——输入一段新闻文本，输出包含 `persons`（人物列表）、`location`（地点）、`time`（时间）、`event`（事件摘要）的 Pydantic 对象。

**验收标准：** 返回对象字段均可直接访问，不需要 parse 字符串；输入不同新闻时字段内容随之变化。

---

### 第 7 章：LangGraph —— 多 Agent 协作

**概念：** LangGraph 把 Agent 工作流建模成有向图（节点 = 行动，边 = 条件流转）。适合需要规划、循环、多角色分工的复杂任务。是构建生产级 Agent 的主流框架。

**关键点：**
- `StateGraph` 定义状态机
- 节点（node）= 一个函数或 agent
- `conditional_edges` 实现条件分支
- `END` 终止节点

**练习题：** 构建一个「Planner + Executor」双 Agent 系统。Planner 把任务分解成步骤列表，Executor 逐步执行每一步，执行完回报 Planner，Planner 判断是否全部完成。

**验收标准：** 输入「帮我调研 Python 最流行的 3 个 web 框架」，系统能自动完成规划（输出步骤列表）和执行（逐步产出结果）两个阶段，graph 的流转清晰可见。

---

### 第 8 章：可观测性与生产化

**概念：** 真实 Agent 需要能调试（LangSmith tracing）、能流式输出、能处理异常、能异步并发。这是从「能跑」到「能用」的最后一步。

**关键点：**
- `LANGCHAIN_TRACING_V2=true` 启用 LangSmith
- `.astream()` 流式异步输出
- `RunnableConfig` 传递运行时配置
- try/except + fallback 优雅降级

**练习题：** 给第 7 章的双 Agent 系统加上：① LangSmith tracing（本地可见完整 trace）；② 流式输出每一步结果到 terminal；③ 某个 tool 抛出异常时 Agent 能优雅降级返回友好提示而不崩溃。

**验收标准：** LangSmith 界面能看到完整 trace 树；terminal 有逐步流式输出；手动触发异常后程序不崩溃，返回降级提示。

---

## 项目文件结构

```
egent/
  chapters/
    ch01_lcel/
      concept.md       ← 概念说明
      example.py       ← 最小可运行示例
      exercise.py      ← 练习题（含 TODO 注释）
      solution.py      ← 参考答案（做完再看）
    ch02_prompt/
    ch03_tools/
    ch04_agent/
    ch05_memory/
    ch06_structured_output/
    ch07_langgraph/
    ch08_production/
  docs/
    superpowers/
      specs/
        2026-06-28-langchain-agent-learning-guide-design.md
  pyproject.toml
```

---

## 依赖清单

| 章节 | 需要新增的依赖 |
|------|--------------|
| 第 1-6 章 | `langchain-openai` |
| 第 7 章 | `langgraph` |
| 第 8 章 | `langsmith` |

安装方式：`uv add <package-name>`

---

## 整体成功标准

完成 8 章后，能独立从零设计并实现一个具备以下能力的 Agent 系统：
- 自主调用多个工具完成多步任务
- 保持多轮对话记忆
- 返回结构化输出
- 多 Agent 协作分工（LangGraph）
- 可观测、可调试、异常安全
