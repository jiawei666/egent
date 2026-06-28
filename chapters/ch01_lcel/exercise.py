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
from re import X
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

model = ChatAnthropic(model="glm-5.2", temperature=0)
parser = StrOutputParser()

# TODO 1: 创建 define_chain，输入 {"word": ...}，输出单词的英文解释（字符串）
define_chain = (
  ChatPromptTemplate.from_messages([("system", "你是一个英语字典专家，解释英语词语或句子的意思, 只需要返回他的英文意思，用英文返回"), ("user", "词语：{word}")]) |
  model |
  parser
)  # 替换这行

# TODO 2: 创建 translate_chain，输入 {"definition": ...}，输出中文翻译（字符串）
translate_chain = (
  ChatPromptTemplate.from_messages([("system", "你是一个翻译专家，把英文翻译成中文"), ("user", "英文：{english}")]) |
  model |
  parser
)  # 替换这行

# TODO 3: 创建 example_chain，输入 {"word": ...}，输出一个英文例句（字符串）
example_chain = (
  ChatPromptTemplate.from_messages([("system", "你是一个英文专家，把给出的英文单词造三个例句"), ("user", "单词：{word}")]) |
  model |
  parser
)  # 替换这行

# TODO 4: 用 RunnablePassthrough.assign 把三个 chain 组合成一个 pipeline
# 最终 lambda 把结果格式化成包含三个字段的字符串
pipeline = (
  RunnablePassthrough.assign(english=define_chain) |
  RunnablePassthrough.assign(translate=translate_chain) |
   RunnablePassthrough.assign(example=example_chain) |
   (lambda x: f"单词：{x["word"]}\n英文解释: {x["english"]}\n中文解释: {x["translate"]}\n例句: {x["example"]}")
)  # 替换这行

if __name__ == "__main__":
    result = pipeline.invoke({"word": "ephemeral"})
    print(result)
    # 验收：输出包含"英文解释:"、"中文翻译:"、"例句:"三行
