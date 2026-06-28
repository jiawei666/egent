"""第 6 章示例：信息提取，返回结构化 Pydantic 对象"""
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

class NewsInfo(BaseModel):
    persons: list[str] = Field(description="Names of people mentioned in the news")
    location: str = Field(description="Primary location where the event occurred")
    time: str = Field(description="When the event occurred (date or time period)")
    event_summary: str = Field(description="One sentence summary of what happened")

model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
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
