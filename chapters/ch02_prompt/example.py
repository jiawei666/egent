"""第 2 章示例：代码评审 bot，返回 Pydantic 结构体"""
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

class CodeReview(BaseModel):
    issues: list[str] = Field(description="List of issues found in the code")
    score: int = Field(description="Code quality score from 0 to 10")
    suggestion: str = Field(description="One key improvement suggestion")

parser = PydanticOutputParser(pydantic_object=CodeReview)
model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

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
    assert isinstance(result.score, int)
    assert isinstance(result.issues, list)
