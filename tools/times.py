from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def get_current_time(tool_input: dict) -> dict:
    timezone: str = tool_input.get("timezone", "Asia/Shanghai")
    print(f"Getting current time for timezone: {timezone}")
    try:
        tz = ZoneInfo(timezone)
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z%z")
        return {
            "content": f"The current time in {timezone} is {now}.",
            "is_error": False,
        }
    except ZoneInfoNotFoundError:
        return {
            "content": (f"Unknown timezone: {timezone!r}. Use an IANA name like 'Asia/Tokyo' or 'UTC'."),
            "is_error": True,
        }


def time_offset(tool_input: dict) -> dict:
    base_time: str = tool_input.get("base_time")
    offset_seconds: int = tool_input.get("offset_seconds")
    try:
        base_dt = datetime.fromisoformat(base_time)
        result = base_dt + timedelta(seconds=offset_seconds)
        return {"content": result.isoformat(), "is_error": False}
    except Exception as e:
        return {"content": f"Invalid base_time format: {e}", "is_error": True}
