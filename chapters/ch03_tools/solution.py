"""第 3 章练习题参考答案"""
from langchain_core.tools import tool

@tool
def search_news(topic: str, max_results: int = 3) -> str:
    """Search for recent news articles on a topic.
    Input: topic (search query string), max_results (number of articles, default 3).
    Returns: a summary string listing article headlines."""
    headlines = [
        f"Breaking: Major {topic} development announced",
        f"Experts weigh in on {topic} trends",
        f"New research reveals {topic} insights",
        f"Government responds to {topic} concerns",
        f"Industry leaders discuss {topic} future",
    ]
    selected = headlines[:max_results]
    return f"{max_results} news articles about '{topic}': {'; '.join(selected)}"

@tool
def calculate_exchange(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between currencies (USD, CNY, EUR).
    Input: amount (float), from_currency (e.g. 'USD'), to_currency (e.g. 'CNY').
    Returns: conversion result string like '100.0 USD = 725.0 CNY'."""
    rates_to_usd = {"USD": 1.0, "CNY": 1 / 7.25, "EUR": 1 / 0.92}
    if from_currency not in rates_to_usd or to_currency not in rates_to_usd:
        return f"Unsupported currency. Supported: USD, CNY, EUR"
    amount_in_usd = amount * rates_to_usd[from_currency]
    result = amount_in_usd / rates_to_usd[to_currency]
    return f"{amount} {from_currency} = {result:.2f} {to_currency}"

if __name__ == "__main__":
    print(search_news.invoke({"topic": "artificial intelligence", "max_results": 2}))
    print(f"Name: {search_news.name}")
    print(f"Description: {search_news.description}\n")

    print(calculate_exchange.invoke({"amount": 100.0, "from_currency": "USD", "to_currency": "CNY"}))
    print(f"Name: {calculate_exchange.name}")
    print(f"Description: {calculate_exchange.description}")
