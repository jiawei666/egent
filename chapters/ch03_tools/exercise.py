"""
第 3 章练习题

目标：创建两个自定义 tools
  1. search_news(topic: str, max_results: int = 3) -> str
     返回格式："{max_results} news articles about {topic}: [article1, article2, ...]"
     用硬编码假数据即可

  2. calculate_exchange(amount: float, from_currency: str, to_currency: str) -> str
     支持 USD/CNY/EUR 之间的换算（用固定汇率即可）
     返回格式："100.0 USD = 725.0 CNY"

要求：
  1. docstring 要清楚描述输入输出格式
  2. tool.invoke({"topic": "AI", "max_results": 2}) 能正确触发
  3. 打印每个 tool 的 name、description、args_schema

验收标准：
  - tool.name 是 "search_news" 和 "calculate_exchange"
  - tool.description 包含参数和返回格式说明
  - tool.invoke(...) 返回预期格式的字符串
"""
from langchain_core.tools import tool

# TODO 1: 定义 search_news tool
# 参数：topic: str, max_results: int = 3
# 用硬编码新闻数据，返回 f"{max_results} news articles about {topic}: ..."


# TODO 2: 定义 calculate_exchange tool
# 参数：amount: float, from_currency: str, to_currency: str
# 汇率：1 USD = 7.25 CNY, 1 USD = 0.92 EUR
# 返回格式：f"{amount} {from_currency} = {result} {to_currency}"


if __name__ == "__main__":
    print(search_news.invoke({"topic": "artificial intelligence", "max_results": 2}))
    print(f"Name: {search_news.name}")
    print(f"Description: {search_news.description}\n")

    print(calculate_exchange.invoke({"amount": 100.0, "from_currency": "USD", "to_currency": "CNY"}))
    print(f"Name: {calculate_exchange.name}")
    print(f"Description: {calculate_exchange.description}")
