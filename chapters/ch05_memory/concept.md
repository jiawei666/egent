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
