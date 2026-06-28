"""第 2 章练习题参考答案"""
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()
model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

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
