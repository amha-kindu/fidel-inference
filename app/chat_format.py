from typing import Dict, List


def render_chat(messages: List[Dict[str, str]]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"[SYSTEM]{content}")
        elif role == "user":
            parts.append(f"[USER]{content}")
        elif role == "assistant":
            parts.append(f"[BOT]{content}")
    parts.append("[BOT]")
    return "".join(parts)
