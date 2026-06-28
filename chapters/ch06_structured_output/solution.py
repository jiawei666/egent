"""第 6 章练习题参考答案"""
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()
model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

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
