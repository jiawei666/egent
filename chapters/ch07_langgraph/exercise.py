"""
第 7 章练习题

目标：构建一个"写作助手" LangGraph 系统
  节点：
    1. outliner —— 给定主题，生成文章大纲（3-4 个要点）
    2. writer —— 逐点展开写作内容
    3. reviewer —— 审阅全文，给出修改建议或通过

  流程：
    START → outliner → writer（循环，每次写一个要点）→ reviewer → END
    如果 reviewer 认为需要修改（state["needs_revision"] = True），回到 writer 重写最后一个要点；
    否则结束。

State 结构：
  - topic: str
  - outline: list[str]
  - current_point: int
  - sections: list[str]（每个要点的写作内容）
  - needs_revision: bool
  - revision_note: str
  - final_article: str

验收标准：
  app.invoke({...}) 能输出一篇结构完整的文章
"""
from dotenv import load_dotenv
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

class State(TypedDict):
    topic: str
    outline: list[str]
    current_point: int
    sections: list[str]
    needs_revision: bool
    revision_note: str
    final_article: str

# TODO 1: 定义 outliner_node（输入 topic，输出 outline 列表和 current_point=0）

# TODO 2: 定义 writer_node（读取 outline[current_point]，写一段内容，追加到 sections，current_point+1）

# TODO 3: 定义 reviewer_node（读取所有 sections，判断质量，返回 needs_revision 和 revision_note）
#          如果通过，把 sections 合并成 final_article

# TODO 4: 定义路由函数
#   writer_router: current_point < len(outline) → "write"，否则 → "review"
#   reviewer_router: needs_revision → "revise"（回 writer），否则 → "finish"（到 END）

# TODO 5: 构建 StateGraph，编译，运行

if __name__ == "__main__":
    pass  # 替换为 app.invoke({...})
