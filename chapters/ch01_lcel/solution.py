"""第 1 章练习题参考答案"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

define_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a dictionary. Give a concise English definition in one sentence."),
        ("human", "Word: {word}"),
    ])
    | model
    | parser
)

translate_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Translate the following English text to Chinese. Only output the translation."),
        ("human", "{definition}"),
    ])
    | model
    | parser
)

example_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Give one natural English example sentence using the word. Only the sentence."),
        ("human", "Word: {word}"),
    ])
    | model
    | parser
)

pipeline = (
    RunnablePassthrough.assign(definition=define_chain)
    | RunnablePassthrough.assign(
        translation=lambda x: translate_chain.invoke({"definition": x["definition"]})
    )
    | RunnablePassthrough.assign(example=example_chain)
    | (lambda x: f"英文解释: {x['definition']}\n中文翻译: {x['translation']}\n例句: {x['example']}")
)

if __name__ == "__main__":
    result = pipeline.invoke({"word": "ephemeral"})
    print(result)
