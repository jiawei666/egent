# 第 3 章：Tools —— 给 LLM 加能力

## @tool 装饰器

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city. Returns temperature and conditions."""
    return f"{city}: 25°C, sunny"

# tool 的关键属性
print(get_weather.name)         # "get_weather"
print(get_weather.description)  # docstring 的内容
print(get_weather.args_schema)  # Pydantic schema，从类型注解自动生成

# 调用 tool
result = get_weather.invoke({"city": "Beijing"})
```

## 为什么 docstring 很重要

LLM 靠 `description` 决定什么时候调用这个 tool。
描述要清楚说明：tool 做什么、接受什么输入、返回什么格式。

差的描述："Gets weather"
好的描述："Get current weather for a city. Input: city name in English. Returns temperature in Celsius and weather conditions."

## 带多参数的 tool

```python
@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search for available flights. Date format: YYYY-MM-DD."""
    return f"Found 3 flights from {origin} to {destination} on {date}"
```
