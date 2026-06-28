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
