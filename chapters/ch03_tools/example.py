"""第 3 章示例：定义和调用自定义 tools"""
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city. Returns temperature in Celsius and conditions.
    Use this when the user asks about weather in a specific location."""
    weather_data = {
        "Beijing": "12°C, cloudy",
        "Shanghai": "18°C, sunny",
        "Shenzhen": "26°C, partly cloudy",
    }
    return weather_data.get(city, f"{city}: 20°C, unknown conditions")

@tool
def get_time(timezone: str) -> str:
    """Get current time for a timezone. Accepts timezone names like 'Asia/Shanghai', 'UTC', 'US/Eastern'.
    Use this when the user asks what time it is in a specific place."""
    from datetime import datetime
    import zoneinfo
    try:
        tz = zoneinfo.ZoneInfo(timezone)
        now = datetime.now(tz)
        return f"Current time in {timezone}: {now.strftime('%H:%M:%S %Z')}"
    except Exception:
        return f"Unknown timezone: {timezone}"

if __name__ == "__main__":
    print(get_weather.invoke({"city": "Beijing"}))
    print(get_time.invoke({"timezone": "Asia/Shanghai"}))

    print(f"\nTool name: {get_weather.name}")
    print(f"Tool description: {get_weather.description}")
    print(f"Args schema: {get_weather.args_schema.model_json_schema()}")
