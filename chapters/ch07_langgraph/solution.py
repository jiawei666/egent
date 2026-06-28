"""第 7 章练习题参考答案"""
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

outline_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are an article planner. Create a 3-4 point outline for an article. Return ONLY a numbered list."),
        ("human", "Topic: {topic}"),
    ]) | model | parser
)

write_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a technical writer. Write a focused 2-3 paragraph section for the given outline point."),
        ("human", "Topic: {topic}\nSection to write: {point}\nRevision note (if any): {note}"),
    ]) | model | parser
)

review_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are an editor. Review the article draft. If it's good enough, say 'APPROVED'. If it needs work, say 'REVISE: ' followed by specific feedback."),
        ("human", "Topic: {topic}\nDraft:\n{draft}"),
    ]) | model | parser
)

def outliner_node(state: State) -> dict:
    print(f"\n[Outliner] Creating outline for: {state['topic']}")
    text = outline_chain.invoke({"topic": state["topic"]})
    points = [l.split(". ", 1)[1] if ". " in l else l for l in text.split("\n") if l.strip() and l.strip()[0].isdigit()]
    print(f"[Outliner] Outline: {points}")
    return {"outline": points, "current_point": 0, "sections": [], "needs_revision": False, "revision_note": ""}

def writer_node(state: State) -> dict:
    point = state["outline"][state["current_point"]]
    print(f"\n[Writer] Writing section: {point}")
    content = write_chain.invoke({"topic": state["topic"], "point": point, "note": state.get("revision_note", "")})
    updated_sections = list(state["sections"])
    if state.get("needs_revision") and updated_sections:
        updated_sections[-1] = content
    else:
        updated_sections.append(content)
    return {"sections": updated_sections, "current_point": state["current_point"] + 1, "needs_revision": False, "revision_note": ""}

def reviewer_node(state: State) -> dict:
    print(f"\n[Reviewer] Reviewing article...")
    draft = "\n\n".join(state["sections"])
    review = review_chain.invoke({"topic": state["topic"], "draft": draft})
    if "APPROVED" in review:
        print("[Reviewer] APPROVED")
        title = f"# {state['topic']}\n\n"
        sections_with_headers = [f"## {state['outline'][i]}\n\n{s}" for i, s in enumerate(state["sections"])]
        return {"needs_revision": False, "final_article": title + "\n\n".join(sections_with_headers)}
    note = review.replace("REVISE:", "").strip()
    print(f"[Reviewer] Needs revision: {note}")
    return {"needs_revision": True, "revision_note": note, "current_point": len(state["sections"]) - 1}

def writer_router(state: State) -> str:
    if state["current_point"] < len(state["outline"]):
        return "write"
    return "review"

def reviewer_router(state: State) -> str:
    return "revise" if state.get("needs_revision") else "finish"

graph = StateGraph(State)
graph.add_node("outliner", outliner_node)
graph.add_node("writer", writer_node)
graph.add_node("reviewer", reviewer_node)
graph.add_edge(START, "outliner")
graph.add_edge("outliner", "writer")
graph.add_conditional_edges("writer", writer_router, {"write": "writer", "review": "reviewer"})
graph.add_conditional_edges("reviewer", reviewer_router, {"revise": "writer", "finish": END})

app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({
        "topic": "Python 异步编程入门",
        "outline": [], "current_point": 0, "sections": [],
        "needs_revision": False, "revision_note": "", "final_article": ""
    })
    print("\n=== Final Article ===")
    print(result["final_article"])
