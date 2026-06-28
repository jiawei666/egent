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
