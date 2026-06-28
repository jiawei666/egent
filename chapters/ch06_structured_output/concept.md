# 第 6 章：Structured Output —— Agent 返回结构化数据

## with_structured_output

让 LLM 返回 Pydantic 对象，底层用 function calling 实现（比 PydanticOutputParser 更可靠）：

```python
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic

class Person(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age")
    city: str = Field(description="City where person lives")

model = ChatAnthropic(model="claude-haiku-4-5-20251001")
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
