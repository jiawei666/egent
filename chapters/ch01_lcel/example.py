"""第 1 章示例：用 LCEL 构建一个翻译 pipeline"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

# Chain 1: 获取英文解释
define_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a dictionary. Give a concise English definition in one sentence."),
        ("human", "Word: {word}"),
    ])
    | model
    | parser
)

# Chain 2: 翻译成中文
translate_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Translate the following English text to Chinese. Only output the translation."),
        ("human", "{definition}"),
    ])
    | model
    | parser
)

# 组合 pipeline：word → definition → translation → 合并输出
pipeline = (
    RunnablePassthrough.assign(definition=define_chain)
    | RunnablePassthrough.assign(
        translation=lambda x: translate_chain.invoke({"definition": x["definition"]})
    )
    | (lambda x: f"英文解释: {x['definition']}\n中文翻译: {x['translation']}")
)

if __name__ == "__main__":
    result = pipeline.invoke({"word": "serendipity"})
    print(result)
