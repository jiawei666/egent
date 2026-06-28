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
